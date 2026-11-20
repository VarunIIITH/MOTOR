"""Class label definitions."""

BEHAVIOR_CLASSES = [
    "overtake", "lane change", "going straight", "left turn",
    "obstruction avoidance", "weaving through traffic",
    "distracted driving", "stop", "right turn", "violation", "u-turn",
]

BEHAVIOR_ABBREV = {
    "distracted driving": "DD", "going straight": "GS", "lane change": "LC",
    "left turn": "LT", "obstruction avoidance": "OA", "overtake": "OT",
    "right turn": "RT", "stop": "Stop", "u-turn": "UT", "violation": "Vio",
    "weaving through traffic": "WTT",
}

LEGALITY_CLASSES = ["legal", "illegal"]
