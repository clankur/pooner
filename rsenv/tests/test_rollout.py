"""E2E rollout test: load Qwen3.5-4B, generate a trajectory with BridgeClient, verify structure."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rsenv import BridgeClient, format_state, rollout_trajectory
from rsenv.state import GameState

MODEL_ID = "Qwen/Qwen3.5-4B"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


@pytest.fixture(scope="module")
def bridge_client():
    client = BridgeClient()
    state = client.start()
    if not state.in_game and state.position == (0, 0):
        client.stop()
        pytest.skip("Game server not reachable at ws://localhost:7780")
    yield client
    client.stop()


def test_bridge_gets_initial_state(bridge_client):
    state = bridge_client.get_state()
    assert state.position != (0, 0), "Should have a real position from the game server"
    print(f"\nLive state:\n{format_state(state)}")


def test_rollout_with_bridge(model_and_tokenizer, bridge_client):
    model, tokenizer, device = model_and_tokenizer
    state = bridge_client.get_state()

    traj = rollout_trajectory(
        model=model,
        tokenizer=tokenizer,
        initial_state=state,
        max_actions=3,
        max_new_tokens=256,
        temperature=0.7,
        device=device,
        client=bridge_client,
    )

    assert len(traj.full_ids) > len(traj.prompt_ids), "Model should generate tokens beyond the prompt"
    assert len(traj.full_ids) == len(traj.generation_mask), "generation_mask must match full_ids length"
    assert traj.generation_mask.sum() > 0, "At least some tokens should be model-generated"

    text = tokenizer.decode(traj.full_ids, skip_special_tokens=False)
    print(f"\n{'='*60}")
    print(f"LIVE TRAJECTORY ({traj.num_actions} actions, {traj.num_valid_actions} valid)")
    print(f"XP gained: {traj.total_xp:.0f} | Reward: {traj.total_reward:.2f}")
    print(f"{'='*60}")
    print(text)
    print(f"\nFinal state:\n{format_state(traj.final_state)}")
