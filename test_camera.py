import cv2
def main() -> None:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera. Try using camera index 1 instead of 0.")
    print("Camera opened successfully. Press Q to close the window.")
    while True:
        success, frame = camera.read()
        if not success:
            raise RuntimeError("Could not read a frame from the camera.")
        cv2.imshow("Camera Test - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()