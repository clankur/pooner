"""Trajectory rollout, prompt formatting, and prompt bank.

The rollout function takes an RSClient (SimClient or BridgeClient) and
drives the plan-then-execute loop. Everything game-specific is in
state.py (data), tools.py (actions), and client.py (execution).
"""

from __future__ import annotations

import base64
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel

if TYPE_CHECKING:
    from transformers import AutoProcessor, PreTrainedModel

from rsenv.client import RSClient
from rsenv.state import ALL_SKILLS, XP_FOR_LEVEL, GameState, Trajectory, xp_to_level
from rsenv.tools import ACTION_PREREQUISITES, TOOL_SCHEMAS, parse_tool_calls

logger = logging.getLogger(__name__)

# ─── Prompt formatting ─────────────────────────────────────────────────────

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text().strip()


def _format_skill_line(skill: str, level: int, xp: int) -> str:
    """Format a single skill with level and XP to next level."""
    if level >= 99:
        return f"{skill} {level}"
    xp_next = XP_FOR_LEVEL[level + 1] - xp
    return f"{skill} {level} ({xp_next} xp to {level + 1})"


def format_state(state: GameState) -> str:
    """Render game state as structured text for the LLM.

    Groups information by category so the model can quickly find what it needs:
    position, status, skills, equipment, inventory, then nearby entities split
    into NPCs, objects, and ground items with distance/reachability metadata.
    """
    if state.world_position != (0, 0):
        pos_str = f"({state.world_position[0]}, {state.world_position[1]})"
    else:
        pos_str = f"({state.position[0]}, {state.position[1]})"

    # Skills: show level + xp-to-next for trained skills
    skill_parts: list[str] = []
    if state.skill_levels:
        for s in ALL_SKILLS:
            lvl = state.skill_levels.get(s, 1)
            if lvl > 1:
                xp = state.skills.get(s, 0)
                skill_parts.append(_format_skill_line(s, lvl, xp))
    else:
        for s in ALL_SKILLS:
            xp = state.skills.get(s, 0)
            if xp > 0:
                lvl = xp_to_level(xp)
                skill_parts.append(_format_skill_line(s, lvl, xp))
    skills_str = ", ".join(skill_parts) if skill_parts else "All level 1"

    # Equipment
    equip_str = ""
    if state.equipment:
        equip_items = [f"{e.name}" for e in state.equipment if e.name]
        if equip_items:
            equip_str = ", ".join(equip_items)

    # Inventory
    if state.inventory_slots:
        inv_items = [f"{s.name} x{s.count}" if s.count > 1 else s.name for s in state.inventory_slots]
        inv_str = ", ".join(inv_items) if inv_items else "Empty"
    else:
        inv_str = ", ".join(f"{item} x{qty}" for item, qty in state.inventory.items()) if state.inventory else "Empty"

    # Nearby NPCs — with combat level, distance, HP if damaged
    npc_parts: list[str] = []
    if state.nearby_npcs:
        for npc in state.nearby_npcs[:8]:
            parts: list[str] = []
            if npc.combat_level > 0:
                parts.append(f"lvl {npc.combat_level}")
            parts.append(f"dist {npc.distance}")
            if npc.max_hp > 0 and npc.hp < npc.max_hp:
                parts.append(f"HP {npc.hp}/{npc.max_hp}")
            if npc.in_combat:
                parts.append("fighting")
            meta = ", ".join(parts)
            opts_str = f" [{', '.join(npc.options)}]" if npc.options else ""
            npc_parts.append(f"  {npc.name} ({meta}){opts_str}")

    # Nearby objects/locs — with distance and interaction options
    # Option labels are actions, not states: "Open" means currently closed, "Close" means currently open
    loc_parts: list[str] = []
    if state.nearby_locs:
        for loc in state.nearby_locs[:8]:
            state_label = ""
            if "Open" in loc.options:
                state_label = ", closed"
            elif "Close" in loc.options:
                state_label = ", open"
            opts_str = f" [{', '.join(loc.options)}]" if loc.options else ""
            loc_parts.append(f"  {loc.name} (dist {loc.distance}{state_label}){opts_str}")

    # Ground items — with distance
    ground_parts: list[str] = []
    if state.ground_items:
        for gi in state.ground_items[:6]:
            qty = f" x{gi.count}" if gi.count > 1 else ""
            ground_parts.append(f"  {gi.name}{qty} (dist {gi.distance})")

    # Fallback if no structured nearby data
    if not npc_parts and not loc_parts and not ground_parts and state.nearby:
        npc_parts = [f"  {name}" for name in state.nearby]

    # Build output
    lines: list[str] = []
    if state.tick > 0:
        lines.append(f"Tick: {state.tick}")
    lines.append(f"Position: {pos_str}")
    lines.append(f"HP: {state.hp}/{state.max_hp}")
    if state.in_combat:
        lines.append("Status: IN COMBAT")
    lines.append(f"Skills: {skills_str}")
    if equip_str:
        lines.append(f"Equipment: {equip_str}")
    lines.append(f"Inventory ({state.inventory_count()}/28): [{inv_str}]")

    if npc_parts:
        lines.append("NPCs:")
        lines.extend(npc_parts)
    if loc_parts:
        lines.append("Objects:")
        lines.extend(loc_parts)
    if ground_parts:
        lines.append("Ground items:")
        lines.extend(ground_parts)
    if not npc_parts and not loc_parts and not ground_parts:
        lines.append("Nearby: Nothing notable")

    return "\n".join(lines)


def format_action_prerequisites(action_name: str) -> str | None:
    """Return prerequisite hint for an action, or None if no prerequisites known."""
    prereqs = ACTION_PREREQUISITES.get(action_name)
    if not prereqs:
        return None
    return f"Requires: {', '.join(prereqs)}"


def build_messages(state: GameState) -> list[dict]:
    system_msg = {"role": "system", "content": load_system_prompt()}
    state_text = format_state(state)

    if state.screenshot:
        b64 = base64.b64encode(state.screenshot).decode()
        user_content = [
            {"type": "image", "image": f"data:image/png;base64,{b64}"},
            {"type": "text", "text": state_text},
        ]
    else:
        user_content = state_text

    return [system_msg, {"role": "user", "content": user_content}]


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
) -> Trajectory:
    """Roll out a plan-then-execute trajectory.

    If client is provided, actions execute through it (sim or live).
    If client is None, a temporary SimClient is created from initial_state.
    Screenshots from the initial state are encoded as image tokens via the processor.
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

    # Processor handles text tokenization + image encoding in one call
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
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

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

    model.eval()
    for _action_idx in range(max_actions):
        input_ids = torch.tensor([all_token_ids], device=device)

        generate_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": stop_ids,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        # Pass image embeddings on the first generation (they correspond to prompt tokens)
        if _action_idx == 0 and pixel_values is not None:
            generate_kwargs["pixel_values"] = pixel_values.to(device)
            if image_grid_thw is not None:
                generate_kwargs["image_grid_thw"] = image_grid_thw.to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids, **generate_kwargs)

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
