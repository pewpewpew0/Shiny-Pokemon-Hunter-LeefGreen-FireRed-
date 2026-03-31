import cv2
import numpy as np


def crop_relative(image, x_min, x_max, y_min, y_max):
    h, w = image.shape[:2]
    x1 = int(w * x_min)
    x2 = int(w * x_max)
    y1 = int(h * y_min)
    y2 = int(h * y_max)
    return image[y1:y2, x1:x2]


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


def orange_panel_mask_bgr(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 20, 150], dtype=np.uint8)
    upper = np.array([25, 120, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def blue_header_mask_bgr(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([95, 80, 80], dtype=np.uint8)
    upper = np.array([120, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def beige_header_mask_bgr(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 20, 150], dtype=np.uint8)
    upper = np.array([40, 120, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def mask_ratio(mask):
    return float(cv2.countNonZero(mask)) / float(mask.shape[0] * mask.shape[1])


def analyze_image(image):
    star_crop = crop_relative(image, 0.405, 0.475, 0.205, 0.295)
    border_crop = crop_relative(image, 0.02, 0.435, 0.08, 0.60)

    yellow = yellow_mask_bgr(star_crop)
    cyan = cyan_mask_bgr(border_crop)
    purple = purple_mask_bgr(border_crop)

    return {
        "yellow_score": mask_ratio(yellow),
        "cyan_score": mask_ratio(cyan),
        "purple_score": mask_ratio(purple),
        "star_crop": star_crop,
        "border_crop": border_crop,
        "yellow_mask": yellow,
        "cyan_mask": cyan,
        "purple_mask": purple,
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


def is_pokemon_info_page(image):
    # Left sprite/info box area
    left_panel = crop_relative(image, 0.02, 0.50, 0.08, 0.62)

    # Right stats/info area
    right_panel = crop_relative(image, 0.52, 0.98, 0.10, 0.72)

    # Top header strip
    top_header = crop_relative(image, 0.00, 1.00, 0.00, 0.12)

    purple_mask = purple_mask_bgr(left_panel)
    orange_mask = orange_panel_mask_bgr(right_panel)
    blue_mask = blue_header_mask_bgr(top_header)
    beige_mask = beige_header_mask_bgr(top_header)

    purple_score = mask_ratio(purple_mask)
    orange_score = mask_ratio(orange_mask)
    blue_score = mask_ratio(blue_mask)
    beige_score = mask_ratio(beige_mask)

    # Tuned from your screenshots:
    # - purple border should be strong on non-shiny info page
    # - right panel peach/orange stripes should be present
    # - top bar has both beige left section and blue right section
    on_info_page = (
        (purple_score > 0.08) and
        (orange_score > 0.10) and
        (blue_score > 0.08) and
        (beige_score > 0.05)
    )

    return on_info_page, {
        "purple_score": purple_score,
        "orange_score": orange_score,
        "blue_score": blue_score,
        "beige_score": beige_score,
        "purple_mask": purple_mask,
        "orange_mask": orange_mask,
        "blue_mask": blue_mask,
        "beige_mask": beige_mask,
        "left_panel": left_panel,
        "right_panel": right_panel,
        "top_header": top_header,
    }


def find_capture_device(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Using capture device index {i}")
            return cap

        cap.release()

    return None


def main():
    cap = find_capture_device()
    if cap is None:
        print("No working capture device found.")
        return

    print("Press 'q' to quit.")
    print("Press 's' to save current frame.")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read frame.")
            break

        frame_count += 1

        on_info_page, page_debug = is_pokemon_info_page(frame)

        display = frame.copy()

        if on_info_page:
            scores = analyze_image(frame)
            decision = classify(scores)

            label = "SHINY" if decision["is_shiny"] else "NOT SHINY"
            color = (0, 0, 255) if decision["is_shiny"] else (0, 255, 0)

            cv2.putText(display, "STATE: INFO PAGE", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(display, label, (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(display, f"yellow={scores['yellow_score']:.4f}", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, f"cyan={scores['cyan_score']:.4f}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, f"purple={scores['purple_score']:.4f}", (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Star Crop", scores["star_crop"])
            cv2.imshow("Border Crop", scores["border_crop"])
            cv2.imshow("Yellow Mask", scores["yellow_mask"])
            cv2.imshow("Cyan Mask", scores["cyan_mask"])
            cv2.imshow("Purple Mask", scores["purple_mask"])

        else:
            cv2.putText(display, "STATE: NOT INFO PAGE", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(display, "SHINY CHECK DISABLED", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, f"page_purple={page_debug['purple_score']:.4f}", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"page_orange={page_debug['orange_score']:.4f}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"page_blue={page_debug['blue_score']:.4f}", (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"page_beige={page_debug['beige_score']:.4f}", (30, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Star Crop", np.zeros((80, 80, 3), dtype=np.uint8))
            cv2.imshow("Border Crop", np.zeros((80, 80, 3), dtype=np.uint8))
            cv2.imshow("Yellow Mask", np.zeros((80, 80), dtype=np.uint8))
            cv2.imshow("Cyan Mask", np.zeros((80, 80), dtype=np.uint8))
            cv2.imshow("Purple Mask", np.zeros((80, 80), dtype=np.uint8))

        cv2.imshow("Live Shiny Check", display)
        cv2.imshow("Info Page Left Panel", page_debug["left_panel"])
        cv2.imshow("Info Page Right Panel", page_debug["right_panel"])
        cv2.imshow("Info Page Header", page_debug["top_header"])
        cv2.imshow("Info Purple Mask", page_debug["purple_mask"])
        cv2.imshow("Info Orange Mask", page_debug["orange_mask"])
        cv2.imshow("Info Blue Mask", page_debug["blue_mask"])
        cv2.imshow("Info Beige Mask", page_debug["beige_mask"])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"live_frame_{frame_count:04d}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()