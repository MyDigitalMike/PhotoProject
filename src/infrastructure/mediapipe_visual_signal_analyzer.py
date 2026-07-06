from __future__ import annotations

from collections import deque
import math
from typing import Any

import cv2
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.hands as mp_hands
import numpy as np

from src.domain.visual_context import VisualContext


class MediaPipeVisualSignalAnalyzer:
    """
    Extracts simple meme-related signals from face and hand landmarks.

    This is not emotion detection.
    This is geometry/rule-based meme-trigger detection.
    """

    MOUTH_OPEN_RATIO_THRESHOLD = 0.075
    MOUTH_WIDE_OPEN_RATIO_THRESHOLD = 0.13
    MOUTH_SMILE_WIDTH_RATIO_THRESHOLD = 0.405
    MOUTH_PUCKERED_WIDTH_RATIO_THRESHOLD = 0.335

    HAND_MOUTH_FACE_RATIO = 0.42
    HAND_FOREHEAD_FACE_RATIO = 0.50
    HAND_CHIN_FACE_RATIO = 0.46
    HAND_TEMPLE_FACE_RATIO = 0.48
    HAND_CHEEK_FACE_RATIO = 0.56
    MIN_HAND_THRESHOLD = 0.075
    MAX_HAND_THRESHOLD = 0.16

    SMOOTHED_SIGNALS = (
        "face_detected",
        "mouth_open",
        "mouth_wide_open",
        "mouth_smile",
        "mouth_puckered",
        "hand_near_mouth",
        "hand_near_forehead",
        "hand_near_chin",
        "hand_near_temple",
        "hand_near_face",
        "hands_near_cheeks",
        "no_hands",
        "any_hands",
        "one_hand",
        "two_hands",
        "thumbs_up",
        "open_palm",
        "one_open_palm",
        "two_open_palms",
    )

    def __init__(
        self,
        signal_history_size: int = 5,
        signal_on_ratio: float = 0.60,
        signal_off_ratio: float = 0.35,
    ) -> None:
        self.signal_history_size = signal_history_size
        self.signal_on_ratio = signal_on_ratio
        self.signal_off_ratio = signal_off_ratio
        self._signal_histories: dict[str, deque[bool]] = {}
        self._stable_signals: dict[str, bool] = {}

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame: np.ndarray) -> VisualContext:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = self.face_mesh.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)

        face_landmarks = self._get_first_face_landmarks(face_results)
        hand_landmarks_list = self._get_hand_landmarks(hand_results)

        face_detected = face_landmarks is not None
        hands_detected = len(hand_landmarks_list)
        no_hands = hands_detected == 0
        any_hands = hands_detected > 0
        one_hand = hands_detected == 1
        two_hands = hands_detected >= 2

        if face_landmarks is None:
            raw_context = VisualContext(
                face_detected=False,
                hands_detected=hands_detected,
                no_hands=no_hands,
                any_hands=any_hands,
                one_hand=one_hand,
                two_hands=two_hands,
                debug={
                    "hands_detected": float(hands_detected),
                },
            )

            return self._smooth_visual_context(raw_context)

        mouth_distance = self._distance_between_face_points(
            face_landmarks,
            13,
            14,
        )
        mouth_open_ratio = self._mouth_open_ratio(face_landmarks)
        mouth_width_ratio = self._mouth_width_ratio(face_landmarks)
        mouth_open = mouth_open_ratio >= self.MOUTH_OPEN_RATIO_THRESHOLD
        mouth_wide_open = mouth_open_ratio >= self.MOUTH_WIDE_OPEN_RATIO_THRESHOLD
        mouth_smile = (
            mouth_width_ratio >= self.MOUTH_SMILE_WIDTH_RATIO_THRESHOLD
            and not mouth_wide_open
        )
        mouth_puckered = (
            not mouth_open
            and mouth_width_ratio <= self.MOUTH_PUCKERED_WIDTH_RATIO_THRESHOLD
        )

        hand_mouth_threshold = self._scaled_face_threshold(
            face_landmarks,
            self.HAND_MOUTH_FACE_RATIO,
        )
        hand_forehead_threshold = self._scaled_face_threshold(
            face_landmarks,
            self.HAND_FOREHEAD_FACE_RATIO,
        )
        hand_chin_threshold = self._scaled_face_threshold(
            face_landmarks,
            self.HAND_CHIN_FACE_RATIO,
        )
        hand_temple_threshold = self._scaled_face_threshold(
            face_landmarks,
            self.HAND_TEMPLE_FACE_RATIO,
        )
        hand_cheek_threshold = self._scaled_face_threshold(
            face_landmarks,
            self.HAND_CHEEK_FACE_RATIO,
        )

        hand_mouth_distance = self._minimum_hand_distance_to_face_point(
            hand_landmarks_list,
            face_landmarks,
            face_index=13,
        )
        hand_forehead_distance = self._minimum_hand_distance_to_face_point(
            hand_landmarks_list,
            face_landmarks,
            face_index=10,
        )
        hand_chin_distance = self._minimum_hand_distance_to_face_point(
            hand_landmarks_list,
            face_landmarks,
            face_index=152,
        )
        hand_temple_distance = self._minimum_hand_distance_to_face_points(
            hand_landmarks_list,
            face_landmarks,
            face_indexes=(127, 356),
        )

        hand_near_mouth = self._is_distance_inside_threshold(
            hand_mouth_distance,
            hand_mouth_threshold,
        )
        hand_near_forehead = self._is_distance_inside_threshold(
            hand_forehead_distance,
            hand_forehead_threshold,
        )
        hand_near_chin = self._is_distance_inside_threshold(
            hand_chin_distance,
            hand_chin_threshold,
        )
        hand_near_temple = self._is_distance_inside_threshold(
            hand_temple_distance,
            hand_temple_threshold,
        )

        hands_near_cheeks = self._are_hands_near_both_cheeks(
            hand_landmarks_list,
            face_landmarks,
            threshold=hand_cheek_threshold,
        )
        hand_near_face = (
            hand_near_mouth
            or hand_near_forehead
            or hand_near_chin
            or hand_near_temple
            or hands_near_cheeks
        )

        thumbs_up_count = sum(
            self._is_thumbs_up(hand_landmarks)
            for hand_landmarks in hand_landmarks_list
        )
        thumbs_up = thumbs_up_count >= 1

        open_palm_count = sum(
            self._is_open_palm(hand_landmarks)
            for hand_landmarks in hand_landmarks_list
        )
        open_palm = open_palm_count >= 1
        one_open_palm = open_palm_count == 1
        two_open_palms = open_palm_count >= 2

        debug = {
            "mouth_distance": mouth_distance,
            "mouth_open_ratio": mouth_open_ratio,
            "mouth_width_ratio": mouth_width_ratio,
            "hands_detected": float(hands_detected),
            "hand_mouth_distance": self._debug_distance(hand_mouth_distance),
            "hand_forehead_distance": self._debug_distance(hand_forehead_distance),
            "hand_chin_distance": self._debug_distance(hand_chin_distance),
            "hand_temple_distance": self._debug_distance(hand_temple_distance),
            "hand_mouth_threshold": hand_mouth_threshold,
            "hand_forehead_threshold": hand_forehead_threshold,
            "hand_chin_threshold": hand_chin_threshold,
            "hand_temple_threshold": hand_temple_threshold,
            "hand_cheek_threshold": hand_cheek_threshold,
            "thumbs_up_count": float(thumbs_up_count),
            "open_palm_count": float(open_palm_count),
            "mouth_wide_open_threshold": self.MOUTH_WIDE_OPEN_RATIO_THRESHOLD,
            "mouth_smile_width_threshold": self.MOUTH_SMILE_WIDTH_RATIO_THRESHOLD,
            "mouth_puckered_width_threshold": (
                self.MOUTH_PUCKERED_WIDTH_RATIO_THRESHOLD
            ),
        }

        raw_context = VisualContext(
            face_detected=face_detected,
            hands_detected=hands_detected,
            mouth_open=mouth_open,
            mouth_closed=not mouth_open,
            mouth_wide_open=mouth_wide_open,
            mouth_smile=mouth_smile,
            mouth_puckered=mouth_puckered,
            hand_near_mouth=hand_near_mouth,
            hand_near_forehead=hand_near_forehead,
            hand_near_chin=hand_near_chin,
            hand_near_temple=hand_near_temple,
            hand_near_face=hand_near_face,
            hands_near_cheeks=hands_near_cheeks,
            no_hands=no_hands,
            any_hands=any_hands,
            one_hand=one_hand,
            two_hands=two_hands,
            thumbs_up=thumbs_up,
            open_palm=open_palm,
            one_open_palm=one_open_palm,
            two_open_palms=two_open_palms,
            debug=debug,
        )

        return self._smooth_visual_context(raw_context)

    @staticmethod
    def _get_first_face_landmarks(face_results: Any) -> Any | None:
        if not face_results.multi_face_landmarks:
            return None

        return face_results.multi_face_landmarks[0].landmark

    @staticmethod
    def _get_hand_landmarks(hand_results: Any) -> list[Any]:
        if not hand_results.multi_hand_landmarks:
            return []

        return [
            hand_landmarks.landmark
            for hand_landmarks in hand_results.multi_hand_landmarks
        ]

    @staticmethod
    def _distance(point_a: Any, point_b: Any) -> float:
        return math.sqrt(
            (point_a.x - point_b.x) ** 2
            + (point_a.y - point_b.y) ** 2
        )

    def _distance_between_face_points(
        self,
        face_landmarks: Any,
        index_a: int,
        index_b: int,
    ) -> float:
        return self._distance(
            face_landmarks[index_a],
            face_landmarks[index_b],
        )

    def _mouth_open_ratio(self, face_landmarks: Any) -> float:
        """
        FaceMesh indexes:
        10 = forehead/top face
        152 = chin
        13 = upper inner lip
        14 = lower inner lip
        """

        mouth_distance = self._distance_between_face_points(
            face_landmarks,
            13,
            14,
        )
        face_height = self._distance_between_face_points(
            face_landmarks,
            10,
            152,
        )

        if face_height <= 0:
            return 0.0

        return mouth_distance / face_height

    def _mouth_width_ratio(self, face_landmarks: Any) -> float:
        """
        FaceMesh indexes:
        61/291 = outer mouth corners
        234/454 = face sides
        """

        mouth_width = self._distance_between_face_points(
            face_landmarks,
            61,
            291,
        )
        face_width = self._distance_between_face_points(
            face_landmarks,
            234,
            454,
        )

        if face_width <= 0:
            return 0.0

        return mouth_width / face_width

    def _scaled_face_threshold(
        self,
        face_landmarks: Any,
        face_ratio: float,
    ) -> float:
        face_width = self._distance_between_face_points(
            face_landmarks,
            234,
            454,
        )

        threshold = face_width * face_ratio

        return min(
            max(threshold, self.MIN_HAND_THRESHOLD),
            self.MAX_HAND_THRESHOLD,
        )

    @staticmethod
    def _is_distance_inside_threshold(
        distance: float | None,
        threshold: float,
    ) -> bool:
        return distance is not None and distance <= threshold

    def _minimum_hand_distance_to_face_point(
        self,
        hand_landmarks_list: list[Any],
        face_landmarks: Any,
        face_index: int,
    ) -> float | None:
        face_point = face_landmarks[face_index]

        important_hand_points = [
            4,   # thumb tip
            8,   # index finger tip
            12,  # middle finger tip
            16,  # ring finger tip
            20,  # pinky tip
            0,   # wrist
        ]

        minimum_distance: float | None = None

        for hand_landmarks in hand_landmarks_list:
            for hand_index in important_hand_points:
                distance = self._distance(
                    hand_landmarks[hand_index],
                    face_point,
                )

                if minimum_distance is None or distance < minimum_distance:
                    minimum_distance = distance

        return minimum_distance

    def _minimum_hand_distance_to_face_points(
        self,
        hand_landmarks_list: list[Any],
        face_landmarks: Any,
        face_indexes: tuple[int, ...],
    ) -> float | None:
        minimum_distance: float | None = None

        for face_index in face_indexes:
            distance = self._minimum_hand_distance_to_face_point(
                hand_landmarks_list,
                face_landmarks,
                face_index=face_index,
            )

            if distance is None:
                continue

            if minimum_distance is None or distance < minimum_distance:
                minimum_distance = distance

        return minimum_distance

    def _are_hands_near_both_cheeks(
        self,
        hand_landmarks_list: list[Any],
        face_landmarks: Any,
        threshold: float,
    ) -> bool:
        if len(hand_landmarks_list) < 2:
            return False

        left_cheek = face_landmarks[234]
        right_cheek = face_landmarks[454]

        left_detected = False
        right_detected = False

        for hand_landmarks in hand_landmarks_list:
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]

            if (
                self._distance(index_tip, left_cheek) <= threshold
                or self._distance(middle_tip, left_cheek) <= threshold
            ):
                left_detected = True

            if (
                self._distance(index_tip, right_cheek) <= threshold
                or self._distance(middle_tip, right_cheek) <= threshold
            ):
                right_detected = True

        return left_detected and right_detected

    def _smooth_visual_context(self, raw_context: VisualContext) -> VisualContext:
        smoothed_values: dict[str, bool] = {}
        debug = dict(raw_context.debug)

        for signal_name in self.SMOOTHED_SIGNALS:
            raw_value = bool(getattr(raw_context, signal_name))
            stable_value, signal_ratio = self._smooth_boolean_signal(
                signal_name,
                raw_value,
            )

            smoothed_values[signal_name] = stable_value
            debug[f"{signal_name}_raw"] = 1.0 if raw_value else 0.0
            debug[f"{signal_name}_smooth_ratio"] = signal_ratio

        return VisualContext(
            face_detected=smoothed_values["face_detected"],
            hands_detected=raw_context.hands_detected,
            mouth_open=smoothed_values["mouth_open"],
            mouth_closed=(
                smoothed_values["face_detected"]
                and not smoothed_values["mouth_open"]
            ),
            mouth_wide_open=smoothed_values["mouth_wide_open"],
            mouth_smile=smoothed_values["mouth_smile"],
            mouth_puckered=smoothed_values["mouth_puckered"],
            hand_near_mouth=smoothed_values["hand_near_mouth"],
            hand_near_forehead=smoothed_values["hand_near_forehead"],
            hand_near_chin=smoothed_values["hand_near_chin"],
            hand_near_temple=smoothed_values["hand_near_temple"],
            hand_near_face=smoothed_values["hand_near_face"],
            hands_near_cheeks=smoothed_values["hands_near_cheeks"],
            no_hands=smoothed_values["no_hands"],
            any_hands=smoothed_values["any_hands"],
            one_hand=smoothed_values["one_hand"],
            two_hands=smoothed_values["two_hands"],
            thumbs_up=smoothed_values["thumbs_up"],
            open_palm=smoothed_values["open_palm"],
            one_open_palm=smoothed_values["one_open_palm"],
            two_open_palms=smoothed_values["two_open_palms"],
            debug=debug,
        )

    def _smooth_boolean_signal(
        self,
        signal_name: str,
        raw_value: bool,
    ) -> tuple[bool, float]:
        history = self._signal_histories.setdefault(
            signal_name,
            deque(maxlen=self.signal_history_size),
        )
        history.append(raw_value)

        signal_ratio = sum(history) / len(history)

        if signal_ratio >= self.signal_on_ratio:
            stable_value = True
        elif signal_ratio <= self.signal_off_ratio:
            stable_value = False
        else:
            stable_value = self._stable_signals.get(signal_name, raw_value)

        self._stable_signals[signal_name] = stable_value

        return stable_value, signal_ratio

    @staticmethod
    def _debug_distance(distance: float | None) -> float:
        if distance is None:
            return -1.0

        return distance

    def _is_thumbs_up(self, hand_landmarks: Any) -> bool:
        """
        Score-based thumbs-up detector.

        In image coordinates:
        - y is smaller when the point is higher on the screen.
        """

        wrist = hand_landmarks[0]
        thumb_mcp = hand_landmarks[2]
        thumb_ip = hand_landmarks[3]
        thumb_tip = hand_landmarks[4]

        index_mcp = hand_landmarks[5]
        index_tip = hand_landmarks[8]
        middle_mcp = hand_landmarks[9]
        middle_tip = hand_landmarks[12]
        ring_mcp = hand_landmarks[13]
        ring_tip = hand_landmarks[16]
        pinky_mcp = hand_landmarks[17]
        pinky_tip = hand_landmarks[20]

        index_pip = hand_landmarks[6]
        middle_pip = hand_landmarks[10]
        ring_pip = hand_landmarks[14]
        pinky_pip = hand_landmarks[18]

        thumb_vertical_distance = thumb_mcp.y - thumb_tip.y
        thumb_horizontal_distance = abs(thumb_tip.x - thumb_mcp.x)

        thumb_points_up = (
            thumb_tip.y < thumb_ip.y
            and thumb_ip.y < thumb_mcp.y
            and thumb_tip.y < wrist.y - 0.10
        )
        thumb_is_vertical = (
            thumb_vertical_distance > 0.065
            and thumb_vertical_distance > thumb_horizontal_distance * 0.70
        )
        thumb_is_above_knuckles = (
            thumb_tip.y < index_mcp.y - 0.025
            and thumb_tip.y < middle_mcp.y - 0.025
        )
        palm_is_upright = (
            index_mcp.y < wrist.y
            and middle_mcp.y < wrist.y
            and ring_mcp.y < wrist.y
            and pinky_mcp.y < wrist.y
        )

        folded_fingers = (
            self._is_finger_folded(index_tip, index_pip, index_mcp),
            self._is_finger_folded(middle_tip, middle_pip, middle_mcp),
            self._is_finger_folded(ring_tip, ring_pip, ring_mcp),
            self._is_finger_folded(pinky_tip, pinky_pip, pinky_mcp),
        )
        folded_count = sum(folded_fingers)

        return (
            thumb_points_up
            and thumb_is_vertical
            and thumb_is_above_knuckles
            and palm_is_upright
            and folded_count >= 3
        )

    @staticmethod
    def _is_finger_folded(
        finger_tip: Any,
        finger_pip: Any,
        finger_mcp: Any,
    ) -> bool:
        tip_not_raised = finger_tip.y >= finger_pip.y - 0.015
        tip_near_palm = abs(finger_tip.y - finger_mcp.y) <= 0.10

        return tip_not_raised or tip_near_palm

    @staticmethod
    def _is_open_palm(hand_landmarks: Any) -> bool:
        index_tip = hand_landmarks[8]
        middle_tip = hand_landmarks[12]
        ring_tip = hand_landmarks[16]
        pinky_tip = hand_landmarks[20]

        index_pip = hand_landmarks[6]
        middle_pip = hand_landmarks[10]
        ring_pip = hand_landmarks[14]
        pinky_pip = hand_landmarks[18]

        fingers_extended = (
            index_tip.y < index_pip.y
            and middle_tip.y < middle_pip.y
            and ring_tip.y < ring_pip.y
            and pinky_tip.y < pinky_pip.y
        )

        return fingers_extended
