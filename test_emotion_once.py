import cv2
from deepface import DeepFace


BACKENDS = [
    "opencv",
    "retinaface",
    "mediapipe",
]


def analyze_with_backend(frame, backend: str) -> None:
    try:
        result = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=backend,
            align=True,
            silent=True,
        )

        if isinstance(result, list):
            result = result[0]

        label = str(result["dominant_emotion"]).lower()
        scores = result["emotion"]

        print(f"\nBackend: {backend}")
        print(f"Detected: {label}")

        for emotion, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            print(f"  {emotion}: {float(score):.5f}%")

    except Exception as error:
        print(f"\nBackend failed: {backend}")
        print(error)


def main() -> None:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open the camera.")

    print("Press SPACE to analyze the current frame.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            raise RuntimeError("Could not read frame from camera.")

        cv2.imshow("Backend Emotion Test - SPACE to analyze", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            print("\n==============================")
            print("Analyzing current frame...")
            print("==============================")

            for backend in BACKENDS:
                analyze_with_backend(frame, backend)

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()