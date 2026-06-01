"""Trajectory rollout, prompt formatting, and prompt bank.

The rollout function takes an RSClient (SimClient or BridgeClient) and
drives the plan-then-execute loop. Everything game-specific is in
state.py (data), tools.py (actions), and client.py (execution).
"""

from __future__ import annotations

import logging
import random
import threading
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
            num_actions += 1
            result = client.execute_action(action_name, action_args)
            state = client.get_state()

            if result.valid:
                num_valid += 1

            append_tool_call(messages, action_name, action_args, think_text)
            append_tool_response(messages, result.observation)
            think_text = ""

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

    reward = 0.0
    reward += total_xp_gained / 100.0
    if num_actions > 0:
        reward += 0.5 * (num_valid / num_actions)
        reward += 0.1 * min(num_actions, max_actions)
    else:
        reward -= 1.0

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
