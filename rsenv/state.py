"""Game state data structures, XP mechanics, and trajectory container."""

from dataclasses import dataclass, field

import torch


def _build_xp_table(max_level: int = 99) -> list[int]:
    """RuneScape XP table: xp(L) = floor(sum(floor(l + 300 * 2^(l/7)) for l in 1..L-1) / 4)"""
    table = [0, 0]
    total = 0
    for lvl in range(1, max_level):
        total += int(lvl + 300 * 2 ** (lvl / 7))
        table.append(total // 4)
    return table


XP_FOR_LEVEL: list[int] = _build_xp_table()

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


@dataclass
class InventorySlot:
    slot: int
    id: int
    name: str
    count: int


@dataclass
class NpcInfo:
    index: int
    name: str
    combat_level: int
    x: int
    z: int
    distance: int
    hp: int
    max_hp: int
    in_combat: bool
    options: list[str]


@dataclass
class LocInfo:
    id: int
    name: str
    x: int
    z: int
    distance: int
    options: list[str]


@dataclass
class GroundItemInfo:
    id: int
    name: str
    count: int
    x: int
    z: int
    distance: int


@dataclass
class ActionResult:
    observation: str
    xp_gained: dict[str, float]
    valid: bool


@dataclass
class GameState:
    tick: int = 0
    position: tuple[int, int] = (0, 0)
    world_position: tuple[int, int] = (0, 0)
    level: int = 0
    skills: dict[str, int] = field(default_factory=dict)
    skill_levels: dict[str, int] = field(default_factory=dict)
    inventory: dict[str, int] = field(default_factory=dict)
    inventory_slots: list[InventorySlot] = field(default_factory=list)
    equipment: list[InventorySlot] = field(default_factory=list)
    nearby: list[str] = field(default_factory=list)
    nearby_npcs: list[NpcInfo] = field(default_factory=list)
    nearby_locs: list[LocInfo] = field(default_factory=list)
    ground_items: list[GroundItemInfo] = field(default_factory=list)
    hp: int = 10
    max_hp: int = 10
    in_combat: bool = False
    in_game: bool = False
    screenshot: bytes | None = None

    def level_for_skill(self, skill: str) -> int:
        if skill in self.skill_levels:
            return self.skill_levels[skill]
        return xp_to_level(self.skills.get(skill, 0))

    def total_level(self) -> int:
        return sum(xp_to_level(xp) for xp in self.skills.values())

    def total_xp(self) -> int:
        return sum(self.skills.values())

    def inventory_count(self) -> int:
        if self.inventory_slots:
            return len(self.inventory_slots)
        return sum(self.inventory.values())

    def copy(self) -> "GameState":
        return GameState(
            tick=self.tick,
            position=self.position,
            world_position=self.world_position,
            level=self.level,
            skills=dict(self.skills),
            skill_levels=dict(self.skill_levels),
            inventory=dict(self.inventory),
            inventory_slots=list(self.inventory_slots),
            equipment=list(self.equipment),
            nearby=list(self.nearby),
            nearby_npcs=list(self.nearby_npcs),
            nearby_locs=list(self.nearby_locs),
            ground_items=list(self.ground_items),
            hp=self.hp,
            max_hp=self.max_hp,
            in_combat=self.in_combat,
            in_game=self.in_game,
            screenshot=self.screenshot,
        )


@dataclass
class Trajectory:
    prompt_ids: torch.Tensor
    full_ids: torch.Tensor
    generation_mask: torch.Tensor
    old_log_probs: torch.Tensor
    total_reward: float
    total_xp: float
    num_actions: int
    num_valid_actions: int
    final_state: GameState
