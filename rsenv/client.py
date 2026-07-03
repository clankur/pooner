"""RSClient interface: uniform API for game interaction.

Two implementations:
  - SimClient: heuristic simulator (offline, no server needed)
  - BridgeClient: live execution via bridge/executor.ts subprocess
"""

import json
import logging
import subprocess
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from rsenv.state import ActionResult, GameState, GroundItemInfo, InventorySlot, LocInfo, NpcInfo
from rsenv.tools import ACTION_PREREQUISITES, LEVEL_REQUIREMENTS, TOOL_NAMES, XP_TABLE

logger = logging.getLogger(__name__)


class RSClient(ABC):
    """Uniform interface for interacting with RuneScape — real or simulated."""

    @abstractmethod
    def get_state(self) -> GameState: ...

    @abstractmethod
    def execute_action(self, name: str, args: BaseModel) -> ActionResult: ...

    @abstractmethod
    def reset(self, initial_state: GameState | None = None) -> GameState: ...


# ─── Heuristic simulator ──────────────────────────────────────────────────


class SimClient(RSClient):
    """Offline heuristic simulator. No game server needed."""

    def __init__(self, initial_state: GameState | None = None) -> None:
        self._state = initial_state.copy() if initial_state else GameState()

    def get_state(self) -> GameState:
        return self._state

    def execute_action(self, name: str, args: BaseModel) -> ActionResult:
        args_dict = args.model_dump(exclude_none=True)
        state = self._state

        if name not in TOOL_NAMES:
            return ActionResult(observation=f"Unknown action: {name}", xp_gained={}, valid=False)

        min_level = LEVEL_REQUIREMENTS.get(name, 1)
        skill_for_action = next(iter(XP_TABLE.get(name, {}).keys()), None)
        if skill_for_action and state.level_for_skill(skill_for_action) < min_level:
            obs = f"Need {skill_for_action} level {min_level} for {name} (current: {state.level_for_skill(skill_for_action)})"
            prereqs = ACTION_PREREQUISITES.get(name)
            if prereqs:
                obs += f" | Requires: {', '.join(prereqs)}"
            return ActionResult(observation=obs, xp_gained={}, valid=False)

        # Item prerequisite checks
        inv_names = set(state.inventory.keys())
        equip_names = {e.name for e in state.equipment} if state.equipment else set()
        all_held = inv_names | equip_names

        if name in ("chopTree", "chopOak"):
            has_axe = any("axe" in item_name.lower() for item_name in all_held)
            if not has_axe:
                return ActionResult(
                    observation=f"Cannot {name}: no axe equipped or in inventory. | Requires: Axe equipped or in inventory",
                    xp_gained={},
                    valid=False,
                )

        if name in ("mineRock", "mineIron"):
            has_pick = any("pickaxe" in item_name.lower() for item_name in all_held)
            if not has_pick:
                return ActionResult(
                    observation=f"Cannot {name}: no pickaxe equipped or in inventory. | Requires: Pickaxe equipped or in inventory",
                    xp_gained={},
                    valid=False,
                )

        if name == "fletchLogs":
            has_knife = any("knife" in item_name.lower() for item_name in all_held)
            has_logs = any("log" in item_name.lower() for item_name in inv_names)
            if not has_knife:
                return ActionResult(
                    observation="Cannot fletch: no Knife in inventory (daggers/swords do not work). | Requires: Knife in inventory, Logs in inventory",
                    xp_gained={},
                    valid=False,
                )
            if not has_logs:
                return ActionResult(
                    observation="Cannot fletch: no logs in inventory. | Requires: Knife in inventory, Logs in inventory",
                    xp_gained={},
                    valid=False,
                )

        xp_gained = dict(XP_TABLE.get(name, {}))
        parts: list[str] = []

        if name in ("chopTree", "chopOak", "mineRock", "mineIron"):
            item = {"chopTree": "Logs", "chopOak": "Oak logs", "mineRock": "Copper ore", "mineIron": "Iron ore"}[name]
            if state.inventory_count() >= 28:
                return ActionResult(observation="Inventory full!", xp_gained={}, valid=False)
            state.inventory[item] = state.inventory.get(item, 0) + 1
            for skill, xp in xp_gained.items():
                state.skills[skill] = state.skills.get(skill, 0) + int(xp)
                parts.append(f"+{xp:.0f} {skill} XP")
            parts.append(f"Got {item}.")

        elif name == "attackNpc":
            for skill, xp in xp_gained.items():
                state.skills[skill] = state.skills.get(skill, 0) + int(xp)
                parts.append(f"+{xp:.0f} {skill} XP")
            parts.append(f"Defeated {args_dict.get('npc_name', 'NPC')}.")

        elif name == "dropInventory":
            item = args_dict.get("item", "")
            if item in state.inventory:
                qty = state.inventory.pop(item)
                parts.append(f"Dropped {item} x{qty}.")
            else:
                return ActionResult(observation=f"No {item} in inventory.", xp_gained={}, valid=False)

        elif name == "walkTo":
            x = args_dict.get("x", state.position[0])
            z = args_dict.get("z", args_dict.get("y", state.position[1]))
            state.position = (x, z)
            state.world_position = (x, z)
            parts.append(f"Walked to ({x}, {z}).")

        elif name == "buryBones":
            if "Bones" in state.inventory:
                state.inventory["Bones"] -= 1
                if state.inventory["Bones"] <= 0:
                    del state.inventory["Bones"]
                for skill, xp in xp_gained.items():
                    state.skills[skill] = state.skills.get(skill, 0) + int(xp)
                    parts.append(f"+{xp:.0f} {skill} XP")
                parts.append("Buried bones.")
            else:
                return ActionResult(observation="No bones in inventory.", xp_gained={}, valid=False)

        else:
            for skill, xp in xp_gained.items():
                state.skills[skill] = state.skills.get(skill, 0) + int(xp)
                parts.append(f"+{xp:.0f} {skill} XP")
            parts.append(f"{name} completed.")

        return ActionResult(observation=" ".join(parts), xp_gained=xp_gained, valid=True)

    def reset(self, initial_state: GameState | None = None) -> GameState:
        if initial_state is not None:
            self._state = initial_state.copy()
        return self._state


# ─── Live bridge client ───────────────────────────────────────────────────


class BridgeClient(RSClient):
    """Live execution via bridge/executor.ts subprocess.

    Communicates via JSON lines over stdin/stdout. The subprocess connects
    to the rs-sdk gateway and translates commands into game actions.
    """

    def __init__(
        self,
        gateway_url: str = "ws://localhost:7780",
        bot_username: str = "grpobot1",
        bot_password: str = "",
        bridge_dir: str | None = None,
    ) -> None:
        self.gateway_url = gateway_url
        self.bot_username = bot_username
        self.bot_password = bot_password
        self.bridge_dir = bridge_dir or str(Path(__file__).parent / "bridge")
        self._process: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._state: GameState = GameState()

    def start(self) -> GameState:
        """Launch the bridge subprocess and wait for initial state."""
        sdk_dir = Path(self.bridge_dir) / "rs-sdk"
        if not sdk_dir.exists():
            setup = Path(self.bridge_dir) / "setup.sh"
            if setup.exists():
                logger.info("Running bridge setup.sh...")
                subprocess.run(["bash", str(setup)], cwd=self.bridge_dir, check=True)

        import os

        # Ensure bun is findable (common install locations)
        path = os.environ.get("PATH", "")
        home = os.path.expanduser("~")
        for bun_dir in [f"{home}/.bun/bin", "/opt/homebrew/bin", "/usr/local/bin"]:
            if bun_dir not in path:
                path = f"{bun_dir}:{path}"

        env = {
            **os.environ,
            "PATH": path,
            "RS_GATEWAY": self.gateway_url,
            "RS_BOT_USERNAME": self.bot_username,
            "RS_BOT_PASSWORD": self.bot_password,
        }

        self._process = subprocess.Popen(
            ["bun", "run", "executor.ts"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.bridge_dir,
            env=env,
            text=True,
            bufsize=1,
        )

        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        response = self._read_response()
        if response and response.get("type") == "state":
            self._state = self._parse_state(response)
        return self._state

    def stop(self) -> None:
        if self._process:
            if self._process.stdin:
                try:
                    self._process.stdin.close()
                except OSError:
                    pass
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def get_state(self) -> GameState:
        self._send_command({"type": "getState"})
        response = self._read_response()
        if response and response.get("type") == "state":
            self._state = self._parse_state(response)
        return self._state

    def execute_action(self, name: str, args: BaseModel) -> ActionResult:
        cmd = {"type": "action", "name": name, "arguments": args.model_dump(exclude_none=True)}
        self._send_command(cmd)
        response = self._read_response()

        if response is None:
            return ActionResult(observation="Bridge communication error.", xp_gained={}, valid=False)
        if response.get("type") == "error":
            return ActionResult(observation=response.get("message", "Unknown error"), xp_gained={}, valid=False)

        # Update cached state
        self._state = self.get_state()
        return ActionResult(
            observation=response.get("observation", ""),
            xp_gained=response.get("xpGained", {}),
            valid=response.get("success", False),
        )

    def reset(self, initial_state: GameState | None = None) -> GameState:
        return self.get_state()

    def send_admin_command(self, command: str) -> ActionResult:
        """Send an admin command (::tele, ::setstat, etc.) via the bridge."""
        cmd = {"type": "action", "name": "adminCommand", "arguments": {"command": command}}
        self._send_command(cmd)
        response = self._read_response()
        if response is None:
            return ActionResult(observation="Bridge communication error.", xp_gained={}, valid=False)
        return ActionResult(
            observation=response.get("observation", response.get("message", "")),
            xp_gained={},
            valid=response.get("success", False),
        )

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _send_command(self, cmd: dict) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("Bridge process not running")
        self._process.stdin.write(json.dumps(cmd) + "\n")
        self._process.stdin.flush()

    def _read_response(self, max_retries: int = 10) -> dict | None:
        """Read one JSON line from stdout, skipping non-JSON lines (e.g. stray console.log from rs-sdk)."""
        if not self._process or not self._process.stdout:
            return None
        for _ in range(max_retries):
            try:
                line = self._process.stdout.readline()
                if not line:
                    return None
                stripped = line.strip()
                if not stripped or not stripped.startswith("{"):
                    logger.debug("Bridge skipping non-JSON line: %s", stripped[:200])
                    continue
                return json.loads(stripped)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Bridge read error: %s", e)
        return None

    def _drain_stderr(self) -> None:
        if not self._process or not self._process.stderr:
            return
        for line in self._process.stderr:
            stripped = line.rstrip()
            if stripped:
                logger.debug("bridge: %s", stripped)

    @staticmethod
    def _parse_state(response: dict) -> GameState:
        data = response.get("data", {})
        pos = data.get("position", {})

        skills_xp: dict[str, int] = {}
        skill_levels: dict[str, int] = {}
        for s in data.get("skills", []):
            name = s.get("name", "")
            skills_xp[name] = s.get("experience", 0)
            skill_levels[name] = s.get("baseLevel", s.get("level", 1))

        inventory: dict[str, int] = {}
        inventory_slots: list[InventorySlot] = []
        for item in data.get("inventory", []):
            name = item.get("name", "")
            count = item.get("count", 1)
            inventory[name] = inventory.get(name, 0) + count
            inventory_slots.append(
                InventorySlot(slot=item.get("slot", 0), id=item.get("id", 0), name=name, count=count)
            )

        equipment = [
            InventorySlot(slot=e.get("slot", 0), id=e.get("id", 0), name=e.get("name", ""), count=e.get("count", 1))
            for e in data.get("equipment", [])
        ]

        nearby_npcs = [
            NpcInfo(
                index=n.get("index", 0),
                name=n.get("name", ""),
                combat_level=n.get("combatLevel", 0),
                x=n.get("x", 0),
                z=n.get("z", 0),
                distance=n.get("distance", 0),
                hp=n.get("hp", 0),
                max_hp=n.get("maxHp", 0),
                in_combat=n.get("inCombat", False),
                options=n.get("options", []),
            )
            for n in data.get("nearbyNpcs", [])
        ]

        nearby_locs = [
            LocInfo(
                id=loc.get("id", 0),
                name=loc.get("name", ""),
                x=loc.get("x", 0),
                z=loc.get("z", 0),
                distance=loc.get("distance", 0),
                options=loc.get("options", []),
            )
            for loc in data.get("nearbyLocs", [])
        ]

        ground_items = [
            GroundItemInfo(
                id=g.get("id", 0),
                name=g.get("name", ""),
                count=g.get("count", 1),
                x=g.get("x", 0),
                z=g.get("z", 0),
                distance=g.get("distance", 0),
            )
            for g in data.get("groundItems", [])
        ]

        nearby_simple = [n.name for n in nearby_npcs] + [loc.name for loc in nearby_locs]

        return GameState(
            tick=data.get("tick", 0),
            position=(pos.get("x", 0), pos.get("z", 0)),
            world_position=(pos.get("worldX", 0), pos.get("worldZ", 0)),
            level=pos.get("level", 0),
            skills=skills_xp,
            skill_levels=skill_levels,
            inventory=inventory,
            inventory_slots=inventory_slots,
            equipment=equipment,
            nearby=nearby_simple,
            nearby_npcs=nearby_npcs,
            nearby_locs=nearby_locs,
            ground_items=ground_items,
            hp=data.get("hp", 10),
            max_hp=data.get("maxHp", 10),
            in_combat=data.get("inCombat", False),
            in_game=data.get("inGame", False),
        )


# ─── Parallel bot pool ────────────────────────────────────────────────────


class BridgeClientPool:
    """Pool of K BridgeClient instances for parallel rollouts.

    Each client connects as a separate named bot (grpobot1..grpobotN).
    Admin commands reset each bot in-place between rollout groups without
    disconnecting, avoiding the reconnect latency of a full reset.
    """

    def __init__(
        self,
        num_clients: int,
        gateway_url: str = "ws://localhost:7780",
        bot_prefix: str = "grpobot",
        bridge_dir: str | None = None,
    ) -> None:
        self.clients: list[BridgeClient] = [
            BridgeClient(
                gateway_url=gateway_url,
                bot_username=f"{bot_prefix}{i}",
                bot_password="",
                bridge_dir=bridge_dir,
            )
            for i in range(1, num_clients + 1)
        ]

    def start_all(self) -> list[GameState]:
        return [client.start() for client in self.clients]

    def stop_all(self) -> None:
        for client in self.clients:
            try:
                client.stop()
            except Exception:
                logger.exception("Failed to stop %s", client.bot_username)

    def reset_all(self, target_state: GameState) -> list[GameState]:
        """Reset all bots to match target_state using admin commands.

        Sends ::tele, ::minme, ::setstat, and ::give commands to each bot
        to synchronize their state with the target without disconnecting.
        """
        return [self._admin_reset(client, target_state) for client in self.clients]

    def _admin_reset(self, client: BridgeClient, target: GameState) -> GameState:
        """Reset a single bot to match target state via admin commands."""
        from rsenv.state import ALL_SKILLS, xp_to_level

        # Teleport to target position using tile coordinates
        x, z = target.position
        client.send_admin_command(f"::tele 0,{x // 64},{z // 64},{x % 64},{z % 64}")

        # Reset all skills to 1 (10 for Hitpoints), then set target levels above baseline
        client.send_admin_command("::minme")
        for skill_name in ALL_SKILLS:
            target_xp = target.skills.get(skill_name, 0)
            if target_xp > 0:
                level = xp_to_level(target_xp)
                baseline = 10 if skill_name == "Hitpoints" else 1
                if level > baseline:
                    client.send_admin_command(f"::setstat {skill_name} {level}")

        # Reset inventory to exactly the target. ::minme/::tele/::give never
        # clear the pack, so items the bot gathered or produced during the
        # previous rollout (logs, ore, cooked food, arrow shafts) otherwise
        # survive the reset, accumulate across rollouts, eventually fill all 28
        # slots, and stall skilling ("Inventory full!"). Wipe first, then hand
        # over the full kit once — no per-tool duplication either, since the
        # pack is empty when we give.
        client.send_admin_command("::clearinv")
        for name, count in target.get_inventory().items():
            client.send_admin_command(f"::give {name} {count}")

        return client.get_state()

    def __len__(self) -> int:
        return len(self.clients)

    def __getitem__(self, idx: int) -> BridgeClient:
        return self.clients[idx]
