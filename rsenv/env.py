"""Trajectory rollout, prompt formatting, and prompt bank.

The rollout function takes an RSClient (SimClient or BridgeClient) and
drives the plan-then-execute loop. Everything game-specific is in
state.py (data), tools.py (actions), and client.py (execution).
"""

from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element, SubElement, indent, tostring

import torch
from pydantic import BaseModel

if TYPE_CHECKING:
    from transformers import AutoProcessor, PreTrainedModel

from rsenv.client import RSClient
from rsenv.state import ALL_SKILLS, XP_FOR_LEVEL, GameState, Trajectory, xp_to_level
from rsenv.tools import TOOL_SCHEMAS, parse_tool_calls

logger = logging.getLogger(__name__)

# ─── Prompt formatting ─────────────────────────────────────────────────────

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text().strip()


def format_state(state: GameState) -> str:
    """Render game state as XML for the LLM."""
    pos = state.world_position if state.world_position != (0, 0) else state.position
    root = Element("game_state")

    if state.tick > 0:
        SubElement(root, "tick").text = str(state.tick)
    SubElement(root, "position", x=str(pos[0]), z=str(pos[1]))
    SubElement(root, "hp", current=str(state.hp), max=str(state.max_hp))
    if state.in_combat:
        SubElement(root, "status").text = "IN COMBAT"

    # Skills
    skills_el = SubElement(root, "skills")
    has_skills = False
    for s in ALL_SKILLS:
        xp = state.skills.get(s, 0)
        lvl = state.skill_levels.get(s) or xp_to_level(xp)
        if lvl > 1:
            has_skills = True
            xp_next = XP_FOR_LEVEL[lvl + 1] - xp if lvl < 99 else 0
            SubElement(skills_el, "skill", name=s, level=str(lvl), xp_to_next=str(xp_next))
    if not has_skills:
        SubElement(skills_el, "skill", name="all", level="1")

    # Equipment
    if state.equipment:
        equip_items = [e for e in state.equipment if e.name]
        if equip_items:
            equip_el = SubElement(root, "equipment")
            for e in equip_items:
                SubElement(equip_el, "item", name=e.name)

    # Inventory
    inv_el = SubElement(root, "inventory", used=str(state.inventory_count()), capacity="28")
    if state.inventory_slots:
        for s in state.inventory_slots:
            attrs = {"name": s.name}
            if s.count > 1:
                attrs["count"] = str(s.count)
            SubElement(inv_el, "item", **attrs)
    elif state.inventory:
        for item, qty in state.inventory.items():
            attrs = {"name": item}
            if qty > 1:
                attrs["count"] = str(qty)
            SubElement(inv_el, "item", **attrs)

    # Nearby NPCs
    npcs = state.nearby_npcs[:8]
    if npcs:
        npcs_el = SubElement(root, "npcs")
        for npc in npcs:
            attrs: dict[str, str] = {"name": npc.name, "distance": str(npc.distance)}
            if npc.combat_level > 0:
                attrs["combat_level"] = str(npc.combat_level)
            if npc.max_hp > 0 and npc.hp < npc.max_hp:
                attrs["hp"] = str(npc.hp)
                attrs["max_hp"] = str(npc.max_hp)
            if npc.in_combat:
                attrs["in_combat"] = "true"
            npc_el = SubElement(npcs_el, "npc", **attrs)
            for opt in npc.options:
                SubElement(npc_el, "option").text = opt

    # Nearby objects/locs
    locs = state.nearby_locs[:8]
    if locs:
        objs_el = SubElement(root, "objects")
        for loc in locs:
            attrs = {"name": loc.name, "distance": str(loc.distance)}
            if "Open" in loc.options:
                attrs["state"] = "closed"
            elif "Close" in loc.options:
                attrs["state"] = "open"
            loc_el = SubElement(objs_el, "object", **attrs)
            for opt in loc.options:
                SubElement(loc_el, "option").text = opt

    # Ground items
    ground = state.ground_items[:6]
    if ground:
        ground_el = SubElement(root, "ground_items")
        for gi in ground:
            attrs = {"name": gi.name, "distance": str(gi.distance)}
            if gi.count > 1:
                attrs["count"] = str(gi.count)
            SubElement(ground_el, "item", **attrs)

    # Fallback for simple nearby list
    if not npcs and not locs and not ground and state.nearby:
        nearby_el = SubElement(root, "nearby")
        for name in state.nearby:
            SubElement(nearby_el, "entity", name=name)

    indent(root)
    return tostring(root, encoding="unicode")


def build_messages(state: GameState) -> list[dict]:
    system_msg = {"role": "system", "content": load_system_prompt()}
    state_text = format_state(state)

    return [system_msg, {"role": "user", "content": state_text}]


def append_tool_call(messages: list[dict], name: str, args: BaseModel, think_text: str = "") -> None:
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
    messages.append({"role": "tool", "content": observation})


# ─── Prompt bank ───────────────────────────────────────────────────────────

STARTING_STATES: list[GameState] = [
    GameState(
        position=(3222, 3218),
        world_position=(3222, 3218),
        skills={"Hitpoints": 1154},
        inventory={"Bronze axe": 1, "Coins": 25},
        nearby=["Tree", "Tree", "Oak tree", "Man", "Woman", "Fishing spot"],
    ),
    GameState(
        position=(3222, 3218),
        world_position=(3222, 3218),
        skills={"Hitpoints": 1154},
        inventory={"Bronze pickaxe": 1, "Coins": 25},
        nearby=["Copper rock", "Tin rock", "Tree", "Man"],
    ),
    GameState(
        position=(3222, 3218),
        world_position=(3222, 3218),
        skills={"Hitpoints": 1154, "Woodcutting": 2107},
        inventory={"Bronze axe": 1},
        nearby=["Tree", "Oak tree", "Oak tree", "Willow tree"],
    ),
    GameState(
        position=(3222, 3218),
        world_position=(3222, 3218),
        skills={"Hitpoints": 1154, "Attack": 388, "Strength": 388},
        inventory={"Bronze sword": 1, "Wooden shield": 1, "Coins": 50},
        nearby=["Chicken", "Chicken", "Cow", "Man", "Tree"],
    ),
    GameState(
        position=(3222, 3218),
        world_position=(3222, 3218),
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


@dataclass(frozen=True)
class SkillingSpawn:
    """A spawn that drops the bot where it can immediately train a skill.

    Each scenario couples a location with the inventory needed to train there —
    a gathering tool (axe/pickaxe/net), combat gear, or raw materials for a
    processing skill — so a freshly-reset bot gains XP on its first action
    instead of wandering. Coordinates and the safety notes below come from
    rs-sdk/learnings/*.md. This replaces independently randomizing location,
    tool, and skill — which spawned bots toolless or far from any resource, so
    no XP could ever flow (exp 217: 0 XP across the whole run, every "valid"
    action was walking or clicking menus).
    """

    name: str
    position: tuple[int, int]
    materials: dict[str, int]  # scenario-specific items, added on top of BASE_TOOLKIT


# Every spawn carries this full kit so the bot must *choose* the right tool for
# what's nearby — an axe is useless at the mine, a pickaxe useless at the trees,
# a net useless against a cow. Being handed only the one correct tool made the
# choice trivial; here the location decides which action actually pays off.
BASE_TOOLKIT: dict[str, int] = {
    "Bronze axe": 1,  # woodcutting
    "Bronze pickaxe": 1,  # mining
    "Small fishing net": 1,  # fishing
    "Bronze sword": 1,  # combat
    "Wooden shield": 1,  # combat
}

# Only verified, level-1-friendly, low-risk scenarios. Deliberately excludes the
# old generic spawns (general store — no resources) and the hazardous/broken
# ones called out in the docs: Lumbridge Swamp mine (interactions fail
# silently), Al Kharid mine (scorpions), Lumbridge Swamp fishing (no small-net
# spots, level 20+ only). Material counts (12) comfortably exceed max_actions so
# a rollout never runs dry mid-skill.
SKILLING_SPAWNS: list[SkillingSpawn] = [
    # Gathering — the right tool is in the kit; the location picks which.
    SkillingSpawn("woodcutting", (3200, 3220), {}),  # Lumbridge trees
    SkillingSpawn("mining", (3285, 3365), {}),  # SE Varrock: copper/tin/iron
    SkillingSpawn("fishing", (3087, 3230), {}),  # Draynor: Net/Bait shrimp
    # Combat — safe, low-level enemies; weapon + shield are in the kit.
    SkillingSpawn("combat_chickens", (3237, 3295), {}),  # very safe
    SkillingSpawn("combat_cows", (3253, 3290), {}),  # safe
    SkillingSpawn("combat_goblins", (3240, 3220), {}),  # mixed
    # Thieving — pickpocket the men in Lumbridge castle courtyard; no tool needed.
    SkillingSpawn("thieving", (3222, 3218), {}),
    # Processing — spawn at the station holding the raw materials.
    SkillingSpawn("cooking", (3211, 3215), {"Raw shrimps": 12}),  # range near Bob's Axes
    SkillingSpawn("fletching", (3200, 3220), {"Knife": 1, "Logs": 12}),  # knife on logs -> arrow shafts
    SkillingSpawn("smithing", (3188, 3421), {"Hammer": 1, "Bronze bar": 12}),  # Varrock west anvil
]


def random_starting_state(rng: random.Random) -> GameState:
    """Pick a skilling scenario: a location plus the full toolkit (and any raw
    materials), so the bot must decide which tool fits what's nearby instead of
    being handed the one right answer.

    Skills are left at baseline (Hitpoints 10, everything else 1) so XP and
    level-ups come fast at low levels — maximizing the learning signal the
    reward depends on.
    """
    spawn = rng.choice(SKILLING_SPAWNS)
    return GameState(
        position=spawn.position,
        world_position=spawn.position,
        skills={"Hitpoints": XP_FOR_LEVEL[10]},
        inventory={**BASE_TOOLKIT, **spawn.materials},
    )


# ─── Reward ───────────────────────────────────────────────────────────────


_ACTION_DECAY = 0.97


def compute_reward(
    total_xp_gained: int,
    num_actions: int,
    num_valid: int,
    initial_skills: dict[str, int],
    final_skills: dict[str, int],
    initial_position: tuple[int, int],
    final_position: tuple[int, int],
    total_gen_tokens: int,
    action_xp_history: list[tuple[str, dict[str, float]]],
    xp_multiplier: int = 1,
) -> tuple[float, int, dict[str, float]]:
    """Compute trajectory reward. Returns (reward, num_level_ups, reward_metrics).

    reward_metrics maps each component to its contribution; the total reward
    is exactly their sum, so wandb plots of the components explain the total.
    """
    reward_metrics: dict[str, float] = {}

    # 1. XP reward (per-action-type decay disabled for now)
    effective_xp = 0.0
    action_counts: dict[str, int] = {}
    for action_name, xp in action_xp_history:
        n = action_counts.get(action_name, 0)
        effective_xp += sum(xp.values()) / xp_multiplier  # * _ACTION_DECAY**n
        action_counts[action_name] = n + 1
    reward_metrics["xp"] = effective_xp / 100.0

    # 2. Level-up bonus
    num_level_ups = sum(
        max(0, xp_to_level(final_skills[s]) - xp_to_level(initial_skills.get(s, 0))) for s in final_skills
    )
    reward_metrics["level_ups"] = 2.0 * num_level_ups

    # 3. Per-action shaping. Exp 172 collapsed to emitting no tool calls:
    # attempting actions netted -1 to -2.3 while doing nothing scored exactly 0,
    # so the policy learned silence in 4 steps. Acting must strictly dominate
    # inaction even before any XP flows, so valid actions earn a bonus and the
    # invalid penalty is small enough that exploration survives it.
    reward_metrics["valid_actions"] = 0.1 * num_valid
    reward_metrics["invalid_actions"] = -0.05 * (num_actions - num_valid)

    # 4. No-tool-call trajectory is the worst outcome — below the worst acting
    # trajectory (all-invalid = -0.5 at max_actions=10), so mixed groups always
    # have advantage pointing toward acting.
    reward_metrics["no_actions"] = -2.0 if num_actions == 0 else 0.0

    # 5. Token efficiency: reward sweet spot, penalize excess
    # if num_actions > 0:
    #     tokens_per_action = total_gen_tokens / num_actions
    #     if tokens_per_action <= 150:
    #         reward_metrics["token_efficiency"] = 0.1 * min(tokens_per_action, 100) / 100
    #     else:
    #         reward_metrics["token_efficiency"] = -0.3 * min((tokens_per_action - 150) / 500, 1.0)

    reward = sum(reward_metrics.values())
    return reward, num_level_ups, reward_metrics


# ─── Trajectory rollout ───────────────────────────────────────────────────


def rollout_trajectory(
    model: PreTrainedModel,
    processor: "AutoProcessor",
    initial_state: GameState,
    max_actions: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    client: RSClient | None = None,
    model_lock: threading.Lock | None = None,
    xp_multiplier: int = 1,
) -> Trajectory:
    """Roll out a plan-then-execute trajectory.

    If client is provided, actions execute through it (sim or live).
    If client is None, a temporary SimClient is created from initial_state.
    """
    if client is not None:
        state = client.reset(initial_state)
    else:
        from rsenv.client import SimClient

        client = SimClient(initial_state)
        state = client.get_state()

    tokenizer = processor.tokenizer
    initial_xp = state.total_xp()
    initial_skills = dict(state.skills)
    initial_position = state.world_position
    messages = build_messages(state)

    inputs = processor.apply_chat_template(
        messages,
        tools=TOOL_SCHEMAS,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_ids = inputs["input_ids"][0]
    prompt_len = len(prompt_ids)

    all_token_ids: list[int] = prompt_ids.tolist()
    gen_mask: list[int] = [0] * prompt_len
    model_log_probs: list[float] = []

    stop_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
        stop_ids.append(im_end_id)

    num_actions = 0
    num_valid = 0
    total_gen_tokens = 0
    action_xp_history: list[tuple[str, dict[str, float]]] = []
    _logged_first = False

    generate_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_ids,
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    model.eval()
    for _action_idx in range(max_actions):
        input_ids = torch.tensor([all_token_ids], device=device)

        if model_lock is not None:
            model_lock.acquire()
        try:
            with torch.no_grad():
                outputs = model.generate(input_ids, **generate_kwargs)
        finally:
            if model_lock is not None:
                model_lock.release()

        new_ids = outputs.sequences[0, len(all_token_ids) :]
        new_text = tokenizer.decode(new_ids, skip_special_tokens=False)
        total_gen_tokens += len(new_ids)

        if not _logged_first:
            logger.info("First generation (%d tokens): %s", len(new_ids), repr(new_text[:1000]))
            _logged_first = True

        for i, score in enumerate(outputs.scores):
            log_probs = torch.log_softmax(score[0], dim=-1)
            token_id = new_ids[i].item()
            model_log_probs.append(log_probs[token_id].item())

        all_token_ids.extend(new_ids.tolist())
        gen_mask.extend([1] * len(new_ids))

        calls = parse_tool_calls(new_text)
        if not calls:
            break

        think_text = ""
        think_start = new_text.find("<think>")
        think_end = new_text.find("</think>")
        if think_start != -1 and think_end != -1:
            think_text = new_text[think_start + len("<think>") : think_end].strip()

        for action_name, action_args in calls:
            if num_actions >= max_actions:
                break
            num_actions += 1
            result = client.execute_action(action_name, action_args)
            state = client.get_state()

            if result.valid:
                num_valid += 1
            action_xp_history.append((action_name, result.xp_gained))

            append_tool_call(messages, action_name, action_args, think_text)
            append_tool_response(messages, result.observation)
            think_text = ""

        if num_actions >= max_actions:
            break

        full_inputs = processor.apply_chat_template(
            messages,
            tools=TOOL_SCHEMAS,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_ids = full_inputs["input_ids"][0]
        env_ids = full_ids[len(all_token_ids) :]

        all_token_ids.extend(env_ids.tolist())
        gen_mask.extend([0] * len(env_ids))

    total_xp_gained = state.total_xp() - initial_xp

    reward, num_level_ups, reward_metrics = compute_reward(
        total_xp_gained=total_xp_gained,
        num_actions=num_actions,
        num_valid=num_valid,
        initial_skills=initial_skills,
        final_skills=state.skills,
        initial_position=initial_position,
        final_position=state.world_position,
        total_gen_tokens=total_gen_tokens,
        action_xp_history=action_xp_history,
        xp_multiplier=xp_multiplier,
    )

    return Trajectory(
        prompt_ids=torch.tensor(prompt_ids.tolist()[:prompt_len]),
        full_ids=torch.tensor(all_token_ids),
        generation_mask=torch.tensor(gen_mask, dtype=torch.float32),
        old_log_probs=torch.tensor(model_log_probs),
        total_reward=reward,
        reward_metrics=reward_metrics,
        total_xp=total_xp_gained,
        num_actions=num_actions,
        num_valid_actions=num_valid,
        num_level_ups=num_level_ups,
        total_gen_tokens=total_gen_tokens,
        final_state=state,
    )
