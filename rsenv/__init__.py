"""rsenv — RuneScape environment for LLM agent training.

Black-box game interface: train.py imports the trajectory rollout and
prompt bank, everything else (simulator, bridge, tools, prompts) is internal.
"""

from rsenv.client import BridgeClient, RSClient, SimClient
from rsenv.env import compute_reward, format_state, load_prompt_bank, rollout_trajectory
from rsenv.state import GameState, Trajectory
from rsenv.tools import TOOL_SCHEMAS, parse_tool_call, parse_tool_calls

__all__ = [
    "BridgeClient",
    "GameState",
    "RSClient",
    "SimClient",
    "TOOL_SCHEMAS",
    "Trajectory",
    "compute_reward",
    "format_state",
    "load_prompt_bank",
    "parse_tool_call",
    "parse_tool_calls",
    "rollout_trajectory",
]
