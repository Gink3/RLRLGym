"""Core constants and action mapping for RLRL Gym."""

ACTION_MOVE_NORTH = 0
ACTION_MOVE_SOUTH = 1
ACTION_MOVE_WEST = 2
ACTION_MOVE_EAST = 3
ACTION_WAIT = 4
ACTION_LOOT = 5
ACTION_EAT = 6
ACTION_PICKUP = 7
ACTION_EQUIP = 8
ACTION_USE = 9
ACTION_INTERACT = 10
ACTION_ATTACK = 11

ACTION_NAMES = {
    ACTION_MOVE_NORTH: "move_north",
    ACTION_MOVE_SOUTH: "move_south",
    ACTION_MOVE_WEST: "move_west",
    ACTION_MOVE_EAST: "move_east",
    ACTION_WAIT: "wait",
    ACTION_LOOT: "loot",
    ACTION_EAT: "eat",
    ACTION_PICKUP: "pickup",
    ACTION_EQUIP: "equip",
    ACTION_USE: "use",
    ACTION_INTERACT: "interact",
    ACTION_ATTACK: "attack",
}

MOVE_DELTAS = {
    ACTION_MOVE_NORTH: (-1, 0),
    ACTION_MOVE_SOUTH: (1, 0),
    ACTION_MOVE_WEST: (0, -1),
    ACTION_MOVE_EAST: (0, 1),
}
