import time

import cv2
from deepface import DeepFace


def analyze_emotion(frame):
    result = DeepFace.analyze(
        img_path=frame,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="opencv",
        silent=True,
    )

    if isinstance(result, list):
        result = result[0]

    dominant_emotion = str(result["dominant_emotion"]).lower()
    emotion_scores = result["emotion"]
    confidence = float(emotion_scores.get(dominant_emotion, 0.0))

    return dominant_emotion, confidence


def main() -> None:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open the camera.")

    current_emotion = "neutral"
    current_confidence = 0.0

    analysis_interval_seconds = 0.75
    last_analysis_time = 0.0

    print("Live emotion detection is running. Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            raise RuntimeError("Could not read a frame from the camera.")

        now = time.time()

        should_analyze = now - last_analysis_time >= analysis_interval_seconds

        if should_analyze:
            try:
                current_emotion, current_confidence = analyze_emotion(frame)
                print(f"Expression: {current_emotion} ({current_confidence:.1f}%)")
            except Exception as error:
                print(f"Emotion analysis failed: {error}")
                current_emotion = "neutral"
                current_confidence = 0.0

            last_analysis_time = now

        text = f"Expression: {current_emotion} ({current_confidence:.1f}%)"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Live Emotion Detection - Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()