"""Content/data loading namespace."""

from ..animals import AnimalDef, load_animals, parse_animals
from ..classes import AgentClass, load_classes
from ..enchantments import EnchantDef, load_enchantments, parse_enchantments
from ..items import ItemCatalog, ItemDef, WeaponDef, load_items, parse_items
from ..monsters import (
    MonsterDef,
    MonsterSpawnEntry,
    load_monster_spawns,
    load_monsters,
    parse_monster_spawns,
    parse_monsters,
)
from ..profiles import AgentProfile, load_profiles
from ..races import AgentRace, load_races
from ..recipes import RecipeDef, load_recipes, parse_recipes
from ..spells import SpellDef, load_spells, parse_spells
from ..statuses import StatusDef, load_statuses, parse_statuses
from ..structures import load_structures_config, parse_structures_config
from ..tiles import load_tileset, parse_tileset

__all__ = [
    "AgentClass",
    "AgentProfile",
    "AgentRace",
    "AnimalDef",
    "EnchantDef",
    "ItemCatalog",
    "ItemDef",
    "MonsterDef",
    "MonsterSpawnEntry",
    "RecipeDef",
    "SpellDef",
    "StatusDef",
    "WeaponDef",
    "load_animals",
    "parse_animals",
    "load_classes",
    "load_enchantments",
    "parse_enchantments",
    "load_items",
    "parse_items",
    "load_monsters",
    "parse_monsters",
    "load_monster_spawns",
    "parse_monster_spawns",
    "load_profiles",
    "load_races",
    "load_recipes",
    "parse_recipes",
    "load_spells",
    "parse_spells",
    "load_statuses",
    "parse_statuses",
    "load_structures_config",
    "parse_structures_config",
    "load_tileset",
    "parse_tileset",
]
