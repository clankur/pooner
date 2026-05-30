"""RuneScape environment: game state, heuristic simulator, trajectory rollout.

Handles the plan-then-execute loop: model generates a plan, then executes it
step-by-step via tool calls, receiving observations between actions.
"""

import json
import random
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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
    "equipItem": {},
}

LEVEL_REQUIREMENTS: dict[str, int] = {
    "chopOak": 15,
    "mineIron": 15,
    "catchFish": 5,
    "pickpocket": 5,
    "castSpell": 3,
    "cookItem": 1,
}

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "chopTree",
            "parameters": {
                "type": "object",
                "properties": {"tree_type": {"type": "string", "enum": ["Tree", "Oak tree"]}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "walkTo",
            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attackNpc",
            "parameters": {"type": "object", "properties": {"npc_name": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dropInventory",
            "parameters": {"type": "object", "properties": {"item": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bankDeposit",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}, "quantity": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bankWithdraw",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}, "quantity": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mineRock",
            "parameters": {
                "type": "object",
                "properties": {"rock_type": {"type": "string", "enum": ["Copper rock", "Tin rock", "Iron rock"]}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "catchFish",
            "parameters": {"type": "object", "properties": {"spot_type": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {"name": "cookItem", "parameters": {"type": "object", "properties": {"item": {"type": "string"}}}},
    },
    {"type": "function", "function": {"name": "buryBones", "parameters": {"type": "object", "properties": {}}}},
    {
        "type": "function",
        "function": {"name": "equipItem", "parameters": {"type": "object", "properties": {"item": {"type": "string"}}}},
    },
    {
        "type": "function",
        "function": {"name": "craftItem", "parameters": {"type": "object", "properties": {"item": {"type": "string"}}}},
    },
    {
        "type": "function",
        "function": {
            "name": "pickpocket",
            "parameters": {"type": "object", "properties": {"npc_name": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "castSpell",
            "parameters": {"type": "object", "properties": {"spell": {"type": "string"}, "target": {"type": "string"}}},
        },
    },
]

TOOL_NAMES: set[str] = {t["function"]["name"] for t in TOOL_SCHEMAS}

XP_FOR_LEVEL: list[int] = [
    0,
    0,
    83,
    174,
    276,
    388,
    512,
    650,
    801,
    969,
    1154,
    1358,
    1584,
    1833,
    2107,
    2411,
    2746,
    3115,
    3523,
    3973,
    4470,
    5018,
    5624,
    6291,
    7028,
    7842,
    8740,
    9730,
    10824,
    12031,
    13363,
    14833,
    16456,
    18247,
    20224,
    22406,
    24815,
    27473,
    30408,
    33648,
    37224,
    41171,
    45529,
    50339,
    55649,
    61512,
    67983,
    75127,
    83014,
    91721,
    101333,
]

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
class GameState:
    position: tuple[int, int]
    skills: dict[str, int]  # skill_name -> xp
    inventory: dict[str, int]  # item_name -> quantity
    nearby: list[str]

    def level(self, skill: str) -> int:
        return xp_to_level(self.skills.get(skill, 0))

    def total_level(self) -> int:
        return sum(xp_to_level(xp) for xp in self.skills.values())

    def total_xp(self) -> int:
        return sum(self.skills.values())

    def inventory_count(self) -> int:
        return sum(self.inventory.values())

    def copy(self) -> "GameState":
        return GameState(
            position=self.position,
            skills=dict(self.skills),
            inventory=dict(self.inventory),
            nearby=list(self.nearby),
        )


@dataclass
class BotAction:
    name: str
    args: dict


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


SYSTEM_PROMPT = """You are a RuneScape agent playing on the LostCity 2004scape server. You have access to game actions as tools. First write a plan inside <plan>...</plan> tags, then execute it step by step. Between each action you will receive an observation with the result. Think inside <think>...</think> tags before each action."""


def format_tools_block() -> str:
    names = sorted(TOOL_NAMES)
    return "Available tools: " + ", ".join(names)


def format_state(state: GameState) -> str:
    skills_str = ", ".join(
        f"{s} {xp_to_level(state.skills.get(s, 0))}" for s in ALL_SKILLS if state.skills.get(s, 0) > 0
    )
    if not skills_str:
        skills_str = "All level 1"

    inv_str = ", ".join(f"{item} x{qty}" for item, qty in state.inventory.items()) if state.inventory else "Empty"
    nearby_str = ", ".join(state.nearby) if state.nearby else "Nothing notable"

    return (
        f"Position: ({state.position[0]}, {state.position[1]})\n"
        f"Skills: {skills_str}\n"
        f"Inventory: [{inv_str}]\n"
        f"Nearby: [{nearby_str}]"
    )


def build_initial_prompt(state: GameState) -> str:
    return f"{SYSTEM_PROMPT}\n\n{format_tools_block()}\n\n[State]\n{format_state(state)}\n\n"


# ─── Tool call parsing ────────────────────────────────────────────────────


def parse_tool_call(text: str) -> BotAction | None:
    start = text.find("<tool_call>")
    if start == -1:
        return None
    end = text.find("</tool_call>", start)
    if end == -1:
        return None
    try:
        call = json.loads(text[start + len("<tool_call>") : end])
        return BotAction(name=call.get("name", ""), args=call.get("arguments", {}))
    except (json.JSONDecodeError, KeyError):
        return None


# ─── Heuristic simulator ──────────────────────────────────────────────────


def simulate_action(state: GameState, action: BotAction) -> ActionResult:
    """Simulate an action's effect on game state, return observation text."""
    name = action.name

    if name not in TOOL_NAMES:
        return ActionResult(
            observation=f"Unknown action: {name}",
            xp_gained={},
            valid=False,
        )

    # Check level requirements
    min_level = LEVEL_REQUIREMENTS.get(name, 1)
    skill_for_action = next(iter(XP_TABLE.get(name, {}).keys()), None)
    if skill_for_action and state.level(skill_for_action) < min_level:
        return ActionResult(
            observation=f"Need {skill_for_action} level {min_level} for {name} (current: {state.level(skill_for_action)})",
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
        parts.append(f"Defeated {action.args.get('npc_name', 'NPC')}.")

    elif name == "dropInventory":
        item = action.args.get("item", "")
        if item in state.inventory:
            qty = state.inventory.pop(item)
            parts.append(f"Dropped {item} x{qty}.")
        else:
            return ActionResult(observation=f"No {item} in inventory.", xp_gained={}, valid=False)

    elif name == "walkTo":
        x = action.args.get("x", state.position[0])
        y = action.args.get("y", state.position[1])
        state.position = (x, y)
        parts.append(f"Walked to ({x}, {y}).")

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
        "equipItem",
        "cookItem",
        "catchFish",
        "craftItem",
        "pickpocket",
        "castSpell",
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
        skills={"Hitpoints": 1154},
        inventory={"Bronze axe": 1, "Coins": 25},
        nearby=["Tree", "Tree", "Oak tree", "Man", "Woman", "Fishing spot"],
    ),
    GameState(
        position=(3222, 3218),
        skills={"Hitpoints": 1154},
        inventory={"Bronze pickaxe": 1, "Coins": 25},
        nearby=["Copper rock", "Tin rock", "Tree", "Man"],
    ),
    GameState(
        position=(3222, 3218),
        skills={"Hitpoints": 1154, "Woodcutting": 2107},
        inventory={"Bronze axe": 1},
        nearby=["Tree", "Oak tree", "Oak tree", "Willow tree"],
    ),
    GameState(
        position=(3222, 3218),
        skills={"Hitpoints": 1154, "Attack": 388, "Strength": 388},
        inventory={"Bronze sword": 1, "Wooden shield": 1, "Coins": 50},
        nearby=["Chicken", "Chicken", "Cow", "Man", "Tree"],
    ),
    GameState(
        position=(3222, 3218),
        skills={"Hitpoints": 1154},
        inventory={"Coins": 100},
        nearby=["Tree", "Man", "Woman", "Fishing spot", "Copper rock"],
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
) -> Trajectory:
    """Roll out a single plan-then-execute trajectory.

    The model generates text (plan + think/action turns). When it emits a
    <tool_call>...</tool_call>, we parse and simulate the action, inject the
    observation, and let the model continue.
    """
    state = initial_state.copy()
    initial_xp = state.total_xp()

    prompt_text = build_initial_prompt(state)
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")[0]
    prompt_len = len(prompt_ids)

    all_token_ids: list[int] = prompt_ids.tolist()
    gen_mask: list[int] = [0] * prompt_len  # prompt tokens are not model-generated
    model_log_probs: list[float] = []

    num_actions = 0
    num_valid = 0

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
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        new_ids = outputs.sequences[0, len(all_token_ids) :]
        new_text = tokenizer.decode(new_ids, skip_special_tokens=False)

        # Compute per-token log-probs from scores
        for i, score in enumerate(outputs.scores):
            log_probs = torch.log_softmax(score[0], dim=-1)
            token_id = new_ids[i].item()
            model_log_probs.append(log_probs[token_id].item())

        all_token_ids.extend(new_ids.tolist())
        gen_mask.extend([1] * len(new_ids))

        action = parse_tool_call(new_text)
        if action is None:
            break

        num_actions += 1
        result = simulate_action(state, action)
        if result.valid:
            num_valid += 1

        obs_text = f"\n[Observation] {result.observation}\n\n"
        obs_ids = tokenizer.encode(obs_text, add_special_tokens=False)
        all_token_ids.extend(obs_ids)
        gen_mask.extend([0] * len(obs_ids))

    total_xp_gained = state.total_xp() - initial_xp

    reward = 0.0
    reward += total_xp_gained / 100.0
    if num_actions > 0:
        reward += 0.5 * (num_valid / num_actions)
    reward += 0.1 * min(num_actions, max_actions)

    return Trajectory(
        prompt_ids=prompt_ids.clone(),
        full_ids=torch.tensor(all_token_ids),
        generation_mask=torch.tensor(gen_mask, dtype=torch.float32),
        old_log_probs=torch.tensor(model_log_probs),
        total_reward=reward,
        total_xp=total_xp_gained,
        num_actions=num_actions,
        num_valid_actions=num_valid,
        final_state=state,
    )
