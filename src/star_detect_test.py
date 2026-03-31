import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = BASE_DIR / "screenshots"
DEBUG_DIR = SCREENSHOT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Add all your real non-shiny screenshots here
NONSHINY_FILES = [
    SCREENSHOT_DIR / "27_pokemon_info.png",
    SCREENSHOT_DIR / "capture_001.png",
    SCREENSHOT_DIR / "capture_002.png",
    SCREENSHOT_DIR / "capture_003.png",
    SCREENSHOT_DIR / "capture_004.png",
]

SHINY_FILE = SCREENSHOT_DIR / "27.5_real_shiny.png"

def crop_relative(image, x_min, x_max, y_min, y_max):
    h, w = image.shape[:2]
    x1 = int(w * x_min)
    x2 = int(w * x_max)
    y1 = int(h * y_min)
    y2 = int(h * y_max)
    return image[y1:y2, x1:x2]

def save_image(name, image):
    out_path = DEBUG_DIR / name
    cv2.imwrite(str(out_path), image)

def yellow_mask_bgr(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 80, 80], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def cyan_mask_bgr(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([75, 50, 50], dtype=np.uint8)
    upper = np.array([110, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def purple_mask_bgr(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([120, 40, 40], dtype=np.uint8)
    upper = np.array([165, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def mask_ratio(mask):
    return float(cv2.countNonZero(mask)) / float(mask.shape[0] * mask.shape[1])

def analyze_image(label, image, save_debug=False):
    star_crop = crop_relative(image, 0.405, 0.475, 0.205, 0.295)
    border_crop = crop_relative(image, 0.02, 0.435, 0.08, 0.60)

    yellow = yellow_mask_bgr(star_crop)
    cyan = cyan_mask_bgr(border_crop)
    purple = purple_mask_bgr(border_crop)

    if save_debug:
        save_image(f"{label}_star_crop.png", star_crop)
        save_image(f"{label}_border_crop.png", border_crop)
        save_image(f"{label}_yellow_mask.png", yellow)
        save_image(f"{label}_cyan_mask.png", cyan)
        save_image(f"{label}_purple_mask.png", purple)

    yellow_score = mask_ratio(yellow)
    cyan_score = mask_ratio(cyan)
    purple_score = mask_ratio(purple)

    return {
        "yellow_score": yellow_score,
        "cyan_score": cyan_score,
        "purple_score": purple_score,
    }

def classify(scores):
    star_detected = scores["yellow_score"] > 0.010
    border_detected = (scores["cyan_score"] > 0.080) and (scores["purple_score"] < 0.050)

    is_shiny = star_detected or border_detected

    return {
        "is_shiny": is_shiny,
        "star_detected": star_detected,
        "border_detected": border_detected,
    }

def main():
    print("=== Testing non-shiny images ===")
    for path in NONSHINY_FILES:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Could not load {path}")
            continue

        scores = analyze_image(path.stem, image, save_debug=False)
        decision = classify(scores)

        print(f"\n{path.name}")
        print(f"  yellow_score = {scores['yellow_score']:.6f}")
        print(f"  cyan_score   = {scores['cyan_score']:.6f}")
        print(f"  purple_score = {scores['purple_score']:.6f}")
        print(f"  star_detected   = {decision['star_detected']}")
        print(f"  border_detected = {decision['border_detected']}")
        print(f"  FINAL shiny?    = {decision['is_shiny']}")

    print("\n=== Testing shiny reference ===")
    shiny = cv2.imread(str(SHINY_FILE))
    if shiny is None:
        print(f"Could not load {SHINY_FILE}")
        return

    scores = analyze_image("shiny_ref", shiny, save_debug=True)
    decision = classify(scores)

    print(f"\n{SHINY_FILE.name}")
    print(f"  yellow_score = {scores['yellow_score']:.6f}")
    print(f"  cyan_score   = {scores['cyan_score']:.6f}")
    print(f"  purple_score = {scores['purple_score']:.6f}")
    print(f"  star_detected   = {decision['star_detected']}")
    print(f"  border_detected = {decision['border_detected']}")
    print(f"  FINAL shiny?    = {decision['is_shiny']}")

if __name__ == "__main__":
    main()