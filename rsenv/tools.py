"""RuneScape BotAction tool definitions as pydantic models.

Single source of truth for tool schemas. Used for:
  - Chat template tool definitions (via to_chat_schemas())
  - Parsing model output (via parse_tool_call())
  - Type-safe action dispatch in the simulator and bridge
"""

import logging
from enum import Enum

from pydantic import BaseModel, Field

from rsenv.tool_parser import extract_tool_call, extract_tool_calls

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

# Items/conditions needed for each action to succeed
ACTION_PREREQUISITES: dict[str, list[str]] = {
    "fletchLogs": ["Knife in inventory", "Logs in inventory"],
    "cookItem": ["Raw food in inventory", "Range or Fire nearby"],
    "smithItem": ["Metal bar in inventory", "Anvil nearby", "Hammer in inventory"],
    "craftItem": ["Needle + Thread in inventory (leather)", "or Chisel (gems)"],
    "chopTree": ["Axe equipped or in inventory"],
    "mineRock": ["Pickaxe equipped or in inventory"],
    "catchFish": ["Net/Rod/Cage in inventory (depends on spot)"],
    "castSpell": ["Required runes in inventory"],
    "buryBones": ["Bones in inventory"],
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

logger = logging.getLogger(__name__)


def _validate(name: str, raw_args: dict) -> tuple[str, BaseModel] | None:
    model_cls = TOOL_MODELS.get(name)
    if model_cls is None:
        logger.info("Unknown tool name: %s", name)
        return None
    try:
        return name, model_cls.model_validate(raw_args)
    except Exception as e:
        logger.info("Tool args validation failed for %s: %s", name, e)
        return None


def parse_tool_call(text: str) -> tuple[str, BaseModel] | None:
    """Parse the first tool call from model output and validate against known tool schemas."""
    result = extract_tool_call(text)
    if result is None:
        return None
    return _validate(*result)


def parse_tool_calls(text: str) -> list[tuple[str, BaseModel]]:
    """Parse all tool calls from model output and validate against known tool schemas."""
    results: list[tuple[str, BaseModel]] = []
    for name, raw_args in extract_tool_calls(text):
        validated = _validate(name, raw_args)
        if validated is not None:
            results.append(validated)
    return results
