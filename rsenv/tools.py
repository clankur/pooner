"""RuneScape BotAction tool definitions as pydantic models.

Single source of truth for tool schemas. Used for:
  - Chat template tool definitions (via to_chat_schemas())
  - Parsing model output (via parse_tool_call())
  - Type-safe action dispatch in the simulator and bridge
"""

import logging
from enum import Enum

from pydantic import BaseModel, Field

# ─── Tool argument models ─────────────────────────────────────────────────


class TreeType(str, Enum):
    tree = "Tree"
    oak = "Oak tree"
    willow = "Willow tree"
    maple = "Maple tree"
    yew = "Yew tree"


class RockType(str, Enum):
    copper = "Copper rock"
    tin = "Tin rock"
    iron = "Iron rock"
    silver = "Silver rock"
    coal = "Coal rock"
    gold = "Gold rock"


class ChopTree(BaseModel):
    """Chop a nearby tree. Walks to it and waits for logs."""

    tree_type: TreeType | None = Field(None, description="Name of the tree to chop")


class WalkTo(BaseModel):
    """Walk to coordinates using pathfinding, auto-opening doors."""

    x: int = Field(..., description="World X coordinate")
    z: int = Field(..., description="World Z coordinate")


class AttackNpc(BaseModel):
    """Attack a nearby NPC, walking to it if needed."""

    npc_name: str = Field(..., description="Name of the NPC to attack")


class DropInventory(BaseModel):
    """Drop an item from inventory."""

    item: str = Field(..., description="Item name to drop")


class OpenBank(BaseModel):
    """Open a nearby bank booth or talk to a banker."""

    pass


class BankDeposit(BaseModel):
    """Deposit an item into the bank. Use quantity=-1 for all."""

    item: str = Field(..., description="Item name to deposit")
    quantity: int = Field(1, description="Amount to deposit (-1 for all)")


class BankWithdraw(BaseModel):
    """Withdraw an item from the bank."""

    item: str = Field(..., description="Item name to withdraw")
    quantity: int = Field(1, description="Amount to withdraw")


class MineRock(BaseModel):
    """Mine a nearby rock. Walks to it and waits for ore."""

    rock_type: RockType | None = Field(None, description="Type of rock to mine")


class CatchFish(BaseModel):
    """Fish at a nearby fishing spot."""

    spot_type: str | None = Field(None, description="Fishing spot name (e.g. 'Fishing spot')")


class CookItem(BaseModel):
    """Cook a raw food item on a nearby range or fire."""

    item: str = Field(..., description="Raw food item to cook (e.g. 'Raw shrimps')")


class BuryBones(BaseModel):
    """Bury bones from inventory for Prayer XP."""

    pass


class EquipItem(BaseModel):
    """Equip a weapon or armour from inventory."""

    item: str = Field(..., description="Item name to equip")


class CraftItem(BaseModel):
    """Craft an item (leather crafting, pottery, etc.)."""

    item: str = Field(..., description="Product to craft (e.g. 'Leather gloves')")


class Pickpocket(BaseModel):
    """Pickpocket a nearby NPC for Thieving XP and loot."""

    npc_name: str = Field(..., description="NPC to pickpocket (e.g. 'Man', 'Farmer')")


class CastSpell(BaseModel):
    """Cast a combat spell on a nearby NPC."""

    spell: str | None = Field(None, description="Spell name (e.g. 'Wind Strike')")
    target: str = Field(..., description="Target NPC name")


class EatFood(BaseModel):
    """Eat food to restore hitpoints."""

    item: str = Field(..., description="Food item to eat")


class PickupItem(BaseModel):
    """Pick up an item from the ground."""

    item: str = Field(..., description="Ground item name to pick up")


class FletchLogs(BaseModel):
    """Fletch logs into bows or arrow shafts using a knife."""

    product: str | None = Field(None, description="Product to fletch (e.g. 'shortbow', 'longbow', 'arrow shafts')")


class SmithItem(BaseModel):
    """Smith a metal bar into an item at an anvil."""

    product: str | None = Field(None, description="Product to smith (e.g. 'dagger', 'axe', 'platebody')")


class TalkTo(BaseModel):
    """Talk to a nearby NPC."""

    npc_name: str = Field(..., description="Name of the NPC to talk to")


# ─── Registry ─────────────────────────────────────────────────────────────

TOOL_MODELS: dict[str, type[BaseModel]] = {
    "chopTree": ChopTree,
    "walkTo": WalkTo,
    "attackNpc": AttackNpc,
    "dropInventory": DropInventory,
    "openBank": OpenBank,
    "bankDeposit": BankDeposit,
    "bankWithdraw": BankWithdraw,
    "mineRock": MineRock,
    "catchFish": CatchFish,
    "cookItem": CookItem,
    "buryBones": BuryBones,
    "equipItem": EquipItem,
    "craftItem": CraftItem,
    "pickpocket": Pickpocket,
    "castSpell": CastSpell,
    "eatFood": EatFood,
    "pickupItem": PickupItem,
    "fletchLogs": FletchLogs,
    "smithItem": SmithItem,
    "talkTo": TalkTo,
}

TOOL_NAMES: set[str] = set(TOOL_MODELS.keys())

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
    "fletchLogs": {"Fletching": 5.0},
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


# ─── Schema generation ────────────────────────────────────────────────────


def to_chat_schemas() -> list[dict]:
    """Generate tool schemas in the format expected by apply_chat_template(tools=...)."""
    schemas = []
    for name, model_cls in TOOL_MODELS.items():
        json_schema = model_cls.model_json_schema()
        # Strip pydantic metadata that the chat template doesn't need
        json_schema.pop("title", None)
        # Convert to the OpenAI-style function format
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": model_cls.__doc__ or "",
                    "parameters": json_schema,
                },
            }
        )
    return schemas


TOOL_SCHEMAS: list[dict] = to_chat_schemas()


# ─── Parsing ──────────────────────────────────────────────────────────────


class ToolCall(BaseModel):
    """Parsed tool call from model output."""

    name: str
    arguments: dict


logger = logging.getLogger(__name__)

_TAG_OPEN = "<tool_call>"
_TAG_CLOSE = "</tool_call>"


def parse_tool_call(text: str) -> tuple[str, BaseModel] | None:
    """Parse a JSON tool call from model output: <tool_call>{"name": "...", "arguments": {...}}</tool_call>"""
    start = text.find(_TAG_OPEN)
    if start == -1:
        return None
    end = text.find(_TAG_CLOSE, start)
    if end == -1:
        return None

    raw = text[start + len(_TAG_OPEN) : end].strip()
    try:
        call = ToolCall.model_validate_json(raw)
    except Exception as e:
        logger.info("Tool call parse failed: %s | raw: %s", e, raw[:200])
        return None

    model_cls = TOOL_MODELS.get(call.name)
    if model_cls is None:
        logger.info("Unknown tool name: %s", call.name)
        return None

    try:
        validated_args = model_cls.model_validate(call.arguments)
    except Exception as e:
        logger.info("Tool args validation failed for %s: %s", call.name, e)
        return None

    return call.name, validated_args
