from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VisualContext:
    face_detected: bool = False
    hands_detected: int = 0

    mouth_open: bool = False
    mouth_closed: bool = False
    mouth_wide_open: bool = False
    hand_near_mouth: bool = False
    hand_near_forehead: bool = False
    hand_near_chin: bool = False
    hand_near_temple: bool = False
    hand_near_face: bool = False
    hands_near_cheeks: bool = False

    no_hands: bool = False
    any_hands: bool = False
    one_hand: bool = False
    two_hands: bool = False
    thumbs_up: bool = False
    open_palm: bool = False
    one_open_palm: bool = False
    two_open_palms: bool = False

    debug: dict[str, float] = field(default_factory=dict)
