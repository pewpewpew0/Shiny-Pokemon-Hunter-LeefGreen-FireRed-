"to run again from your project root, use this command: ./.venv/bin/python src/capture_test.py"
import cv2
from pathlib import Path

SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)

def find_capture_device(max_index: int = 10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Using capture device index {i}")
            return cap, i

        cap.release()

    return None, None

def main():
    cap, index = find_capture_device()
    if cap is None:
        print("No working capture device found.")
        return

    print("Press 's' to save a screenshot.")
    print("Press 'q' to quit.")

    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read frame from capture device.")
            break

        cv2.imshow("Switch Capture Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            screenshot_count += 1
            filename = SCREENSHOT_DIR / f"capture_{screenshot_count:03d}.png"
            cv2.imwrite(str(filename), frame)
            print(f"Saved {filename}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()