"""rsenv — RuneScape environment for LLM agent training.

Black-box game interface: train.py imports the trajectory rollout and
prompt bank, everything else (simulator, bridge, tools, prompts) is internal.
"""

from rsenv.client import BridgeClient, BridgeClientPool, RSClient, SimClient
from rsenv.env import compute_reward, format_state, load_prompt_bank, random_starting_state, rollout_trajectory
from rsenv.generation import GenerationService
from rsenv.state import GameState, Trajectory
from rsenv.tools import TOOL_SCHEMAS, parse_tool_call, parse_tool_calls

__all__ = [
    "BridgeClient",
    "BridgeClientPool",
    "GameState",
    "GenerationService",
    "RSClient",
    "SimClient",
    "TOOL_SCHEMAS",
    "Trajectory",
    "compute_reward",
    "format_state",
    "load_prompt_bank",
    "parse_tool_call",
    "parse_tool_calls",
    "random_starting_state",
    "rollout_trajectory",
]
