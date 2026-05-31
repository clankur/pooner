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
from rsenv.tools import LEVEL_REQUIREMENTS, TOOL_NAMES, XP_TABLE

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
            return ActionResult(
                observation=f"Need {skill_for_action} level {min_level} for {name} (current: {state.level_for_skill(skill_for_action)})",
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

        env = {
            **os.environ,
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
        # Don't disconnect/reconnect — just refresh state. The bot stays in-game.
        return self.get_state()
        return self._state

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
