from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VisualContext:
    face_detected: bool = False
    hands_detected: int = 0

    mouth_open: bool = False
    mouth_closed: bool = False
    mouth_wide_open: bool = False
    mouth_smile: bool = False
    mouth_puckered: bool = False
    eyes_wide: bool = False
    eyes_squint: bool = False
    eyes_closed: bool = False
    wink: bool = False
    eyebrows_raised: bool = False
    head_tilt_left: bool = False
    head_tilt_right: bool = False
    looking_left: bool = False
    looking_right: bool = False
    looking_up: bool = False
    looking_down: bool = False
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
    peace_sign: bool = False
    finger_pointing: bool = False
    fist: bool = False
    hand_wave: bool = False
    open_palm: bool = False
    one_open_palm: bool = False
    two_open_palms: bool = False

    debug: dict[str, float] = field(default_factory=dict)
