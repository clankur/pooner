"""Trajectory rollout, prompt formatting, and prompt bank.

The rollout function takes an RSClient (SimClient or BridgeClient) and
drives the plan-then-execute loop. Everything game-specific is in
state.py (data), tools.py (actions), and client.py (execution).
"""

import logging
import random
from pathlib import Path

import torch
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from rsenv.client import RSClient
from rsenv.state import ALL_SKILLS, GameState, Trajectory, xp_to_level
from rsenv.tools import TOOL_SCHEMAS, parse_tool_call

logger = logging.getLogger(__name__)

# ─── Prompt formatting ─────────────────────────────────────────────────────

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text().strip()


def format_state(state: GameState) -> str:
    """Render game state as markdown for the LLM."""
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
        inv_items = [f"{s.name} x{s.count}" if s.count > 1 else s.name for s in state.inventory_slots]
        inv_str = ", ".join(inv_items) if inv_items else "Empty"
    else:
        inv_str = ", ".join(f"{item} x{qty}" for item, qty in state.inventory.items()) if state.inventory else "Empty"

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
        f"HP: {state.hp}/{state.max_hp}",
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
    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": format_state(state)},
    ]


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
    tokenizer: PreTrainedTokenizerBase,
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
    """
    if client is not None:
        state = client.reset(initial_state)
    else:
        from rsenv.client import SimClient

        client = SimClient(initial_state)
        state = client.get_state()

    initial_xp = state.total_xp()
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    messages = build_messages(state) if has_chat_template else None

    if has_chat_template:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMAS,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    else:
        prompt_text = f"{load_system_prompt()}\n\n{format_state(state)}\n\n"

    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)[0]
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

        result = client.execute_action(action_name, action_args)
        state = client.get_state()

        if result.valid:
            num_valid += 1

        obs_content = result.observation

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
            full_ids = tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False)[0]
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
