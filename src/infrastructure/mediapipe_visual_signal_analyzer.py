from __future__ import annotations

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

    def __init__(self) -> None:
 

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

        if face_landmarks is None:
            return VisualContext(
                face_detected=False,
                hands_detected=hands_detected,
            )

        mouth_open = self._is_mouth_open(face_landmarks)

        hand_near_mouth = self._is_any_hand_near_face_point(
            hand_landmarks_list,
            face_landmarks,
            face_index=13,
            threshold=0.10,
        )

        hand_near_forehead = self._is_any_hand_near_face_point(
            hand_landmarks_list,
            face_landmarks,
            face_index=10,
            threshold=0.12,
        )

        hand_near_chin = self._is_any_hand_near_face_point(
            hand_landmarks_list,
            face_landmarks,
            face_index=152,
            threshold=0.11,
        )

        hands_near_cheeks = self._are_hands_near_both_cheeks(
            hand_landmarks_list,
            face_landmarks,
        )

        thumbs_up = any(
            self._is_thumbs_up(hand_landmarks)
            for hand_landmarks in hand_landmarks_list
        )

        open_palm = any(
            self._is_open_palm(hand_landmarks)
            for hand_landmarks in hand_landmarks_list
        )

        debug = {
            "mouth_distance": self._distance_between_face_points(
                face_landmarks,
                13,
                14,
            ),
            "hands_detected": float(hands_detected),
        }

        return VisualContext(
            face_detected=face_detected,
            hands_detected=hands_detected,
            mouth_open=mouth_open,
            hand_near_mouth=hand_near_mouth,
            hand_near_forehead=hand_near_forehead,
            hand_near_chin=hand_near_chin,
            hands_near_cheeks=hands_near_cheeks,
            thumbs_up=thumbs_up,
            open_palm=open_palm,
            debug=debug,
        )

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

    def _is_mouth_open(self, face_landmarks: Any) -> bool:
        """
        FaceMesh indexes:
        13 = upper inner lip
        14 = lower inner lip
        """

        mouth_distance = self._distance_between_face_points(
            face_landmarks,
            13,
            14,
        )

        return mouth_distance >= 0.030

    def _is_any_hand_near_face_point(
        self,
        hand_landmarks_list: list[Any],
        face_landmarks: Any,
        face_index: int,
        threshold: float,
    ) -> bool:
        face_point = face_landmarks[face_index]

        important_hand_points = [
            4,   # thumb tip
            8,   # index finger tip
            12,  # middle finger tip
            16,  # ring finger tip
            20,  # pinky tip
            0,   # wrist
        ]

        for hand_landmarks in hand_landmarks_list:
            for hand_index in important_hand_points:
                distance = self._distance(
                    hand_landmarks[hand_index],
                    face_point,
                )

                if distance <= threshold:
                    return True

        return False

    def _are_hands_near_both_cheeks(
        self,
        hand_landmarks_list: list[Any],
        face_landmarks: Any,
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
                self._distance(index_tip, left_cheek) <= 0.14
                or self._distance(middle_tip, left_cheek) <= 0.14
            ):
                left_detected = True

            if (
                self._distance(index_tip, right_cheek) <= 0.14
                or self._distance(middle_tip, right_cheek) <= 0.14
            ):
                right_detected = True

        return left_detected and right_detected

    @staticmethod
    def _is_thumbs_up(hand_landmarks: Any) -> bool:
        """
        Rough thumbs-up detector.

        In image coordinates:
        - y is smaller when the point is higher on the screen.
        """

        wrist = hand_landmarks[0]
        thumb_tip = hand_landmarks[4]

        index_tip = hand_landmarks[8]
        middle_tip = hand_landmarks[12]
        ring_tip = hand_landmarks[16]
        pinky_tip = hand_landmarks[20]

        index_pip = hand_landmarks[6]
        middle_pip = hand_landmarks[10]
        ring_pip = hand_landmarks[14]
        pinky_pip = hand_landmarks[18]

        thumb_is_up = thumb_tip.y < wrist.y - 0.12

        fingers_are_folded = (
            index_tip.y > index_pip.y
            and middle_tip.y > middle_pip.y
            and ring_tip.y > ring_pip.y
            and pinky_tip.y > pinky_pip.y
        )

        return thumb_is_up and fingers_are_folded

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