"""RuneScape environment: game state, heuristic simulator, trajectory rollout.

Handles the plan-then-execute loop: model generates a plan, then executes it
step-by-step via tool calls, receiving observations between actions.

Supports two backends:
  - Heuristic simulator (offline training, no live server needed)
  - GameBridge (live execution via bridge/executor.ts subprocess)
"""

import json
import logging
import random
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from tools import TOOL_NAMES, TOOL_SCHEMAS, parse_tool_call

logger = logging.getLogger(__name__)

# ─── Game knowledge ────────────────────────────────────────────────────────

XP_TABLE: dict[str, dict[str, float]] = {
    "chopTree": {"Woodcutting": 25.0},
    "chopOak": {"Woodcutting": 37.5},
    "mineRock": {"Mining": 17.5},
    "mineIron": {"Mining": 35.0},
    "attackNpc": {"Attack": 20.0, "Hitpoints": 13.3},
    "castSpell": {"Magic": 11.5},
    "cookItem": {"Cooking": 30.0},
    "catchFish": {"Fishing": 20.0},
    "craftItem": {"Crafting": 22.5},
    "buryBones": {"Prayer": 4.5},
    "pickpocket": {"Thieving": 8.0},
    "walkTo": {},
    "dropInventory": {},
    "bankDeposit": {},
    "bankWithdraw": {},
    "openBank": {},
    "equipItem": {},
    "eatFood": {},
    "pickupItem": {},
    "fletchLogs": {"Fletching": 5.0},
    "talkTo": {},
    "smithItem": {"Smithing": 12.5},
}

LEVEL_REQUIREMENTS: dict[str, int] = {
    "chopOak": 15,
    "mineIron": 15,
    "catchFish": 5,
    "pickpocket": 5,
    "castSpell": 3,
    "cookItem": 1,
}


def _build_xp_table(max_level: int = 99) -> list[int]:
    """RuneScape XP table: xp(L) = floor(sum(floor(l + 300 * 2^(l/7)) for l in 1..L-1) / 4)"""
    table = [0, 0]
    total = 0
    for lvl in range(1, max_level):
        total += int(lvl + 300 * 2 ** (lvl / 7))
        table.append(total // 4)
    return table


XP_FOR_LEVEL: list[int] = _build_xp_table()

ALL_SKILLS: list[str] = [
    "Attack",
    "Strength",
    "Defence",
    "Hitpoints",
    "Ranged",
    "Prayer",
    "Magic",
    "Cooking",
    "Woodcutting",
    "Fletching",
    "Fishing",
    "Firemaking",
    "Crafting",
    "Smithing",
    "Mining",
    "Herblore",
    "Agility",
    "Thieving",
    "Slayer",
    "Runecrafting",
]


def xp_to_level(xp: int) -> int:
    for lvl in range(len(XP_FOR_LEVEL) - 1, 0, -1):
        if xp >= XP_FOR_LEVEL[lvl]:
            return lvl
    return 1


# ─── Game state ────────────────────────────────────────────────────────────


@dataclass
class InventorySlot:
    slot: int
    id: int
    name: str
    count: int


@dataclass
class NpcInfo:
    index: int
    name: str
    combat_level: int
    x: int
    z: int
    distance: int
    hp: int
    max_hp: int
    in_combat: bool
    options: list[str]


@dataclass
class LocInfo:
    id: int
    name: str
    x: int
    z: int
    distance: int
    options: list[str]


@dataclass
class GroundItemInfo:
    id: int
    name: str
    count: int
    x: int
    z: int
    distance: int


@dataclass
class GameState:
    """Game state matching the rs-sdk BotWorldState structure.

    When running with the heuristic simulator, only position/skills/inventory/nearby
    are populated. The bridge populates the richer fields (tick, world coords,
    inventory slots, NPC details, etc.).
    """

    tick: int = 0
    position: tuple[int, int] = (0, 0)
    world_position: tuple[int, int] = (0, 0)
    level: int = 0  # map plane (0=ground, 1=upstairs, etc.)
    skills: dict[str, int] = field(default_factory=dict)  # skill_name -> xp
    skill_levels: dict[str, int] = field(default_factory=dict)  # skill_name -> base_level (from server)
    inventory: dict[str, int] = field(default_factory=dict)  # item_name -> quantity (heuristic mode)
    inventory_slots: list[InventorySlot] = field(default_factory=list)  # full slot info (bridge mode)
    equipment: list[InventorySlot] = field(default_factory=list)
    nearby: list[str] = field(default_factory=list)  # simplified names (heuristic mode)
    nearby_npcs: list[NpcInfo] = field(default_factory=list)
    nearby_locs: list[LocInfo] = field(default_factory=list)
    ground_items: list[GroundItemInfo] = field(default_factory=list)
    hp: int = 10
    max_hp: int = 10
    in_combat: bool = False
    in_game: bool = False

    def level_for_skill(self, skill: str) -> int:
        # Prefer server-reported level when available
        if skill in self.skill_levels:
            return self.skill_levels[skill]
        return xp_to_level(self.skills.get(skill, 0))

    def total_level(self) -> int:
        return sum(xp_to_level(xp) for xp in self.skills.values())

    def total_xp(self) -> int:
        return sum(self.skills.values())

    def inventory_count(self) -> int:
        if self.inventory_slots:
            return len(self.inventory_slots)
        return sum(self.inventory.values())

    def copy(self) -> "GameState":
        return GameState(
            tick=self.tick,
            position=self.position,
            world_position=self.world_position,
            level=self.level,
            skills=dict(self.skills),
            skill_levels=dict(self.skill_levels),
            inventory=dict(self.inventory),
            inventory_slots=list(self.inventory_slots),
            equipment=list(self.equipment),
            nearby=list(self.nearby),
            nearby_npcs=list(self.nearby_npcs),
            nearby_locs=list(self.nearby_locs),
            ground_items=list(self.ground_items),
            hp=self.hp,
            max_hp=self.max_hp,
            in_combat=self.in_combat,
            in_game=self.in_game,
        )


@dataclass
class ActionResult:
    observation: str
    xp_gained: dict[str, float]
    valid: bool


@dataclass
class Trajectory:
    prompt_ids: torch.Tensor  # (prompt_len,)
    full_ids: torch.Tensor  # (total_len,) — prompt + all generated + observations
    generation_mask: torch.Tensor  # (total_len,) — 1 for model tokens, 0 for env tokens
    old_log_probs: torch.Tensor  # (num_model_tokens,) — log-probs under policy at generation time
    total_reward: float
    total_xp: float
    num_actions: int
    num_valid_actions: int
    final_state: GameState


# ─── Prompt formatting ─────────────────────────────────────────────────────

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text().strip()


def format_state(state: GameState) -> str:
    """Render game state as markdown the LLM reads.

    Uses the richer data fields when populated (bridge mode), falls back
    to the simpler fields for heuristic mode.
    """
    if state.world_position != (0, 0):
        pos_str = f"({state.world_position[0]}, {state.world_position[1]})"
    else:
        pos_str = f"({state.position[0]}, {state.position[1]})"

    if state.skill_levels:
        skills_str = ", ".join(
            f"{s} {state.skill_levels[s]}" for s in ALL_SKILLS if s in state.skill_levels and state.skill_levels[s] > 1
        )
    else:
        skills_str = ", ".join(
            f"{s} {xp_to_level(state.skills.get(s, 0))}" for s in ALL_SKILLS if state.skills.get(s, 0) > 0
        )
    if not skills_str:
        skills_str = "All level 1"

    if state.inventory_slots:
        inv_items: list[str] = []
        for slot in state.inventory_slots:
            if slot.count > 1:
                inv_items.append(f"{slot.name} x{slot.count}")
            else:
                inv_items.append(slot.name)
        inv_str = ", ".join(inv_items) if inv_items else "Empty"
    else:
        inv_str = ", ".join(f"{item} x{qty}" for item, qty in state.inventory.items()) if state.inventory else "Empty"

    hp_str = f"{state.hp}/{state.max_hp}"

    nearby_parts: list[str] = []
    if state.nearby_npcs:
        for npc in state.nearby_npcs[:10]:
            combat_str = f" (lvl {npc.combat_level})" if npc.combat_level > 0 else ""
            hp_npc = f" HP:{npc.hp}/{npc.max_hp}" if npc.max_hp > 0 else ""
            nearby_parts.append(f"{npc.name}{combat_str}{hp_npc}")
    if state.nearby_locs:
        for loc in state.nearby_locs[:10]:
            opts = f" [{', '.join(loc.options)}]" if loc.options else ""
            nearby_parts.append(f"{loc.name}{opts}")
    if state.ground_items:
        for gi in state.ground_items[:5]:
            qty = f" x{gi.count}" if gi.count > 1 else ""
            nearby_parts.append(f"[Ground] {gi.name}{qty}")
    if not nearby_parts and state.nearby:
        nearby_parts = list(state.nearby)
    nearby_str = ", ".join(nearby_parts) if nearby_parts else "Nothing notable"

    lines = [
        f"Position: {pos_str}",
        f"HP: {hp_str}",
        f"Skills: {skills_str}",
        f"Inventory ({state.inventory_count()}/28): [{inv_str}]",
        f"Nearby: [{nearby_str}]",
    ]

    if state.in_combat:
        lines.insert(2, "Status: IN COMBAT")

    if state.tick > 0:
        lines.insert(0, f"Tick: {state.tick}")

    return "\n".join(lines)


def build_messages(state: GameState) -> list[dict]:
    """Build the initial chat messages list for apply_chat_template."""
    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": format_state(state)},
    ]


def append_tool_call(messages: list[dict], name: str, args: BaseModel, think_text: str = "") -> None:
    """Append an assistant tool-call turn to the messages list."""
    content = f"<think>\n{think_text}\n</think>\n\n" if think_text else ""
    messages.append(
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": args.model_dump(exclude_none=True)}}
            ],
        }
    )


def append_tool_response(messages: list[dict], observation: str) -> None:
    """Append a tool response turn to the messages list."""
    messages.append({"role": "tool", "content": observation})


# ─── Tool call parsing ────────────────────────────────────────────────────


# ─── GameBridge (live execution via subprocess) ───────────────────────────


class GameBridge:
    """Manages a bridge/executor.ts subprocess for live game interaction.

    Communicates via JSON lines over stdin/stdout. The subprocess connects
    to the rs-sdk gateway and translates our commands into game actions.
    """

    def __init__(
        self,
        gateway_url: str = "ws://localhost:7780",
        bot_username: str = "grpo_agent",
        bot_password: str = "",
        bridge_dir: str | None = None,
    ) -> None:
        self.gateway_url = gateway_url
        self.bot_username = bot_username
        self.bot_password = bot_password
        if bridge_dir is None:
            self.bridge_dir = str(Path(__file__).parent / "bridge")
        else:
            self.bridge_dir = bridge_dir
        self._process: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> GameState | None:
        """Launch the bridge subprocess and wait for initial state."""
        env = {
            "RS_GATEWAY": self.gateway_url,
            "RS_BOT_USERNAME": self.bot_username,
            "RS_BOT_PASSWORD": self.bot_password,
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        }

        self._process = subprocess.Popen(
            ["bun", "run", "executor.ts"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.bridge_dir,
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Drain stderr in background so the subprocess doesn't block
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        # Read initial state response
        response = self._read_response()
        if response and response.get("type") == "state":
            return self._parse_state(response)
        elif response:
            logger.warning("Bridge startup response: %s", response)
        return None

    def stop(self) -> None:
        """Shut down the bridge subprocess."""
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

    def get_state(self) -> GameState | None:
        """Request current game state from the bridge."""
        self._send_command({"type": "getState"})
        response = self._read_response()
        if response and response.get("type") == "state":
            return self._parse_state(response)
        return None

    def execute_action(self, name: str, args: dict) -> ActionResult:
        """Send an action to the bridge and return the result."""
        cmd: dict = {
            "type": "action",
            "name": name,
            "arguments": args,
        }
        self._send_command(cmd)
        response = self._read_response()

        if response is None:
            return ActionResult(
                observation="Bridge communication error.",
                xp_gained={},
                valid=False,
            )

        if response.get("type") == "error":
            return ActionResult(
                observation=response.get("message", "Unknown error"),
                xp_gained={},
                valid=False,
            )

        # type == "result"
        return ActionResult(
            observation=response.get("observation", ""),
            xp_gained=response.get("xpGained", {}),
            valid=response.get("success", False),
        )

    def reset(self) -> GameState | None:
        """Reset the bridge (disconnect/reconnect to get fresh state)."""
        self._send_command({"type": "reset"})
        response = self._read_response()
        if response and response.get("type") == "state":
            return self._parse_state(response)
        return None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _send_command(self, cmd: dict) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("Bridge process not running")
        line = json.dumps(cmd) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

    def _read_response(self, timeout_sec: float = 60.0) -> dict | None:
        """Read one JSON line from stdout. Returns None on EOF or error."""
        if not self._process or not self._process.stdout:
            return None
        try:
            line = self._process.stdout.readline()
            if not line:
                return None
            return json.loads(line.strip())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Bridge read error: %s", e)
            return None

    def _drain_stderr(self) -> None:
        """Forward subprocess stderr to Python logging."""
        if not self._process or not self._process.stderr:
            return
        for line in self._process.stderr:
            stripped = line.rstrip()
            if stripped:
                logger.debug("bridge: %s", stripped)

    @staticmethod
    def _parse_state(response: dict) -> GameState:
        """Convert a bridge state response into a GameState."""
        data = response.get("data", {})
        pos = data.get("position", {})
        skills_data = data.get("skills", [])
        inv_data = data.get("inventory", [])
        equip_data = data.get("equipment", [])
        npcs_data = data.get("nearbyNpcs", [])
        locs_data = data.get("nearbyLocs", [])
        ground_data = data.get("groundItems", [])

        # Build skill dicts from server data
        skills_xp: dict[str, int] = {}
        skill_levels: dict[str, int] = {}
        for s in skills_data:
            name = s.get("name", "")
            skills_xp[name] = s.get("experience", 0)
            skill_levels[name] = s.get("baseLevel", s.get("level", 1))

        # Build inventory dicts (both formats)
        inventory: dict[str, int] = {}
        inventory_slots: list[InventorySlot] = []
        for item in inv_data:
            name = item.get("name", "")
            count = item.get("count", 1)
            inventory[name] = inventory.get(name, 0) + count
            inventory_slots.append(
                InventorySlot(
                    slot=item.get("slot", 0),
                    id=item.get("id", 0),
                    name=name,
                    count=count,
                )
            )

        equipment = [
            InventorySlot(
                slot=e.get("slot", 0),
                id=e.get("id", 0),
                name=e.get("name", ""),
                count=e.get("count", 1),
            )
            for e in equip_data
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
            for n in npcs_data
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
            for loc in locs_data
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
            for g in ground_data
        ]

        # Also build the simplified nearby list for format_state fallback
        nearby_simple: list[str] = []
        for npc in nearby_npcs:
            nearby_simple.append(npc.name)
        for loc in nearby_locs:
            nearby_simple.append(loc.name)

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


def execute_real_action(bridge: GameBridge, name: str, args: BaseModel) -> ActionResult:
    """Execute an action via the GameBridge and return the result."""
    return bridge.execute_action(name, args.model_dump(exclude_none=True))


# ─── Heuristic simulator ──────────────────────────────────────────────────


def simulate_action(state: GameState, name: str, args: BaseModel) -> ActionResult:
    """Simulate an action's effect on game state, return observation text."""
    args_dict = args.model_dump(exclude_none=True)

    if name not in TOOL_NAMES:
        return ActionResult(
            observation=f"Unknown action: {name}",
            xp_gained={},
            valid=False,
        )

    # Check level requirements
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

    elif name in (
        "bankDeposit",
        "bankWithdraw",
        "openBank",
        "equipItem",
        "cookItem",
        "catchFish",
        "craftItem",
        "pickpocket",
        "castSpell",
        "eatFood",
        "pickupItem",
        "fletchLogs",
        "smithItem",
        "talkTo",
    ):
        for skill, xp in xp_gained.items():
            state.skills[skill] = state.skills.get(skill, 0) + int(xp)
            parts.append(f"+{xp:.0f} {skill} XP")
        parts.append(f"{name} completed.")

    else:
        parts.append(f"{name} completed.")

    obs = " ".join(parts)
    return ActionResult(observation=obs, xp_gained=xp_gained, valid=True)


# ─── Prompt bank ───────────────────────────────────────────────────────────


STARTING_STATES: list[GameState] = [
    GameState(
        position=(3222, 3218),
        world_position=(3222, 3218),
        skills={"Hitpoints": 1154},
        inventory={"Bronze axe": 1, "Coins": 25},
        nearby=["Tree", "Tree", "Oak tree", "Man", "Woman", "Fishing spot"],
        hp=10,
        max_hp=10,
    ),
    GameState(
        position=(3222, 3218),
        world_position=(3222, 3218),
        skills={"Hitpoints": 1154},
        inventory={"Bronze pickaxe": 1, "Coins": 25},
        nearby=["Copper rock", "Tin rock", "Tree", "Man"],
        hp=10,
        max_hp=10,
    ),
    GameState(
        position=(3222, 3218),
        world_position=(3222, 3218),
        skills={"Hitpoints": 1154, "Woodcutting": 2107},
        inventory={"Bronze axe": 1},
        nearby=["Tree", "Oak tree", "Oak tree", "Willow tree"],
        hp=10,
        max_hp=10,
    ),
    GameState(
        position=(3222, 3218),
        world_position=(3222, 3218),
        skills={"Hitpoints": 1154, "Attack": 388, "Strength": 388},
        inventory={"Bronze sword": 1, "Wooden shield": 1, "Coins": 50},
        nearby=["Chicken", "Chicken", "Cow", "Man", "Tree"],
        hp=10,
        max_hp=10,
    ),
    GameState(
        position=(3222, 3218),
        world_position=(3222, 3218),
        skills={"Hitpoints": 1154},
        inventory={"Coins": 100},
        nearby=["Tree", "Man", "Woman", "Fishing spot", "Copper rock"],
        hp=10,
        max_hp=10,
    ),
]


def load_prompt_bank(seed: int = 42) -> list[GameState]:
    rng = random.Random(seed)
    bank = [s.copy() for s in STARTING_STATES]
    rng.shuffle(bank)
    return bank


# ─── Trajectory rollout ───────────────────────────────────────────────────


def rollout_trajectory(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    initial_state: GameState,
    max_actions: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    bridge: GameBridge | None = None,
) -> Trajectory:
    """Roll out a single plan-then-execute trajectory using Qwen3's chat template.

    Uses apply_chat_template for proper special token formatting. The model
    generates <think> reasoning and <tool_call> actions. Tool responses are
    injected via the template's tool response format.
    """
    if bridge is not None:
        state = bridge.get_state()
        if state is None:
            state = initial_state.copy()
            logger.warning("Bridge returned no state, falling back to initial state")
    else:
        state = initial_state.copy()

    initial_xp = state.total_xp()
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    messages = build_messages(state) if has_chat_template else None

    if has_chat_template:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMAS,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        prompt_text = f"{load_system_prompt()}\n\n{format_state(state)}\n\n"

    # add_special_tokens=False: apply_chat_template already includes BOS/special tokens
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)[0]
    prompt_len = len(prompt_ids)

    all_token_ids: list[int] = prompt_ids.tolist()
    gen_mask: list[int] = [0] * prompt_len
    model_log_probs: list[float] = []

    # Build stop token list: eos + <|im_end|> for chat models
    stop_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
        stop_ids.append(im_end_id)

    num_actions = 0
    num_valid = 0
    _logged_first = False

    model.eval()
    for _action_idx in range(max_actions):
        input_ids = torch.tensor([all_token_ids], device=device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_ids,
                return_dict_in_generate=True,
                output_scores=True,
            )

        new_ids = outputs.sequences[0, len(all_token_ids) :]
        new_text = tokenizer.decode(new_ids, skip_special_tokens=False)

        if not _logged_first:
            logger.info("First generation (%d tokens): %s", len(new_ids), repr(new_text[:1000]))
            _logged_first = True

        for i, score in enumerate(outputs.scores):
            log_probs = torch.log_softmax(score[0], dim=-1)
            token_id = new_ids[i].item()
            model_log_probs.append(log_probs[token_id].item())

        all_token_ids.extend(new_ids.tolist())
        gen_mask.extend([1] * len(new_ids))

        parsed = parse_tool_call(new_text)
        if parsed is None:
            break

        action_name, action_args = parsed
        num_actions += 1

        think_text = ""
        think_start = new_text.find("<think>")
        think_end = new_text.find("</think>")
        if think_start != -1 and think_end != -1:
            think_text = new_text[think_start + len("<think>") : think_end].strip()

        if bridge is not None:
            result = execute_real_action(bridge, action_name, action_args)
            new_state = bridge.get_state()
            if new_state is not None:
                state = new_state
        else:
            result = simulate_action(state, action_name, action_args)

        if result.valid:
            num_valid += 1

        obs_content = result.observation
        if bridge is not None:
            obs_content += f"\n\n{format_state(state)}"

        if has_chat_template:
            append_tool_call(messages, action_name, action_args, think_text)
            append_tool_response(messages, obs_content)

            full_text = tokenizer.apply_chat_template(
                messages,
                tools=TOOL_SCHEMAS,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            full_ids = tokenizer.encode(full_text, return_tensors="pt")[0]
            env_ids = full_ids[len(all_token_ids) :]
        else:
            obs_text = f"\n[Observation] {obs_content}\n\n"
            env_ids = torch.tensor(tokenizer.encode(obs_text, add_special_tokens=False))

        all_token_ids.extend(env_ids.tolist())
        gen_mask.extend([0] * len(env_ids))

    total_xp_gained = state.total_xp() - initial_xp

    reward = 0.0
    reward += total_xp_gained / 100.0
    if num_actions > 0:
        reward += 0.5 * (num_valid / num_actions)
    reward += 0.1 * min(num_actions, max_actions)

    return Trajectory(
        prompt_ids=torch.tensor(prompt_ids.tolist()[:prompt_len]),
        full_ids=torch.tensor(all_token_ids),
        generation_mask=torch.tensor(gen_mask, dtype=torch.float32),
        old_log_probs=torch.tensor(model_log_probs),
        total_reward=reward,
        total_xp=total_xp_gained,
        num_actions=num_actions,
        num_valid_actions=num_valid,
        final_state=state,
    )
