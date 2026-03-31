import cv2
import numpy as np
import random
import time


# ============================================================
# Vision helpers
# ============================================================

def crop_relative(image, x_min, x_max, y_min, y_max):
    h, w = image.shape[:2]
    x1 = int(w * x_min)
    x2 = int(w * x_max)
    y1 = int(h * y_min)
    y2 = int(h * y_max)
    return image[y1:y2, x1:x2]


def mask_ratio(mask):
    return float(cv2.countNonZero(mask)) / float(mask.shape[0] * mask.shape[1])


def hsv_mask(image, lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.array(lower, dtype=np.uint8),
        np.array(upper, dtype=np.uint8)
    )


# ---------- shiny detection masks ----------

def yellow_mask_bgr(image):
    return hsv_mask(image, [15, 80, 80], [40, 255, 255])


def cyan_mask_bgr(image):
    return hsv_mask(image, [75, 50, 50], [110, 255, 255])


def purple_mask_bgr(image):
    return hsv_mask(image, [120, 40, 40], [165, 255, 255])


def orange_panel_mask_bgr(image):
    return hsv_mask(image, [5, 20, 150], [25, 120, 255])


def blue_header_mask_bgr(image):
    return hsv_mask(image, [95, 80, 80], [120, 255, 255])


def beige_header_mask_bgr(image):
    return hsv_mask(image, [15, 20, 150], [40, 120, 255])


# ---------- final info page detection ----------

def is_pokemon_info_page(image):
    left_panel = crop_relative(image, 0.02, 0.50, 0.08, 0.62)
    right_panel = crop_relative(image, 0.52, 0.98, 0.10, 0.72)
    top_header = crop_relative(image, 0.00, 1.00, 0.00, 0.12)

    purple_score = mask_ratio(purple_mask_bgr(left_panel))
    orange_score = mask_ratio(orange_panel_mask_bgr(right_panel))
    blue_score = mask_ratio(blue_header_mask_bgr(top_header))
    beige_score = mask_ratio(beige_header_mask_bgr(top_header))

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
    }


def analyze_shiny(image):
    star_crop = crop_relative(image, 0.405, 0.475, 0.205, 0.295)
    border_crop = crop_relative(image, 0.02, 0.435, 0.08, 0.60)

    yellow_score = mask_ratio(yellow_mask_bgr(star_crop))
    cyan_score = mask_ratio(cyan_mask_bgr(border_crop))
    purple_score = mask_ratio(purple_mask_bgr(border_crop))

    star_detected = yellow_score > 0.010
    border_detected = (cyan_score > 0.080) and (purple_score < 0.050)
    is_shiny = star_detected or border_detected

    return {
        "is_shiny": is_shiny,
        "star_detected": star_detected,
        "border_detected": border_detected,
        "yellow_score": yellow_score,
        "cyan_score": cyan_score,
        "purple_score": purple_score,
    }


# ---------- state detectors matching your PDF ----------

def is_press_start_page(image):
    top_crop = crop_relative(image, 0.00, 1.00, 0.00, 0.20)
    bottom_crop = crop_relative(image, 0.00, 1.00, 0.78, 1.00)

    green_mask_top = hsv_mask(top_crop, [40, 80, 80], [85, 255, 255])
    green_mask_bottom = hsv_mask(bottom_crop, [40, 80, 80], [85, 255, 255])

    top_score = mask_ratio(green_mask_top)
    bottom_score = mask_ratio(green_mask_bottom)

    return (top_score > 0.10) and (bottom_score > 0.08)


def is_continue_page(image):
    crop = crop_relative(image, 0.06, 0.62, 0.05, 0.70)
    blue_mask = hsv_mask(crop, [95, 40, 80], [125, 255, 255])
    return mask_ratio(blue_mask) > 0.04


def is_yes_prompt_page(image):
    # catches the "YES / NO" prompt used for taking Charmander and nickname
    crop = crop_relative(image, 0.68, 0.93, 0.28, 0.66)
    blue_mask = hsv_mask(crop, [95, 40, 80], [125, 255, 255])
    return mask_ratio(blue_mask) > 0.04


def is_options_screen(image):
    crop = crop_relative(image, 0.68, 0.95, 0.10, 0.68)
    blue_mask = hsv_mask(crop, [95, 40, 80], [125, 255, 255])
    return mask_ratio(blue_mask) > 0.04


def is_summary_screen(image):
    # detect the teal party screen with the popup menu
    left_crop = crop_relative(image, 0.00, 0.65, 0.10, 0.90)
    teal_mask = hsv_mask(left_crop, [75, 40, 80], [110, 255, 255])

    menu_crop = crop_relative(image, 0.62, 0.95, 0.38, 0.80)
    menu_blue = hsv_mask(menu_crop, [95, 40, 80], [125, 255, 255])

    return (mask_ratio(teal_mask) > 0.10) and (mask_ratio(menu_blue) > 0.03)


# ============================================================
# Capture
# ============================================================

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


# ============================================================
# Controller placeholder layer
# ============================================================

CONTROLLER_ENABLED = False

def jitter(lo, hi):
    return random.uniform(lo, hi)


def press(button_name, hold=0.08, after=0.12):
    print(f"[ACTION] PRESS {button_name} hold={hold:.2f}s after={after:.2f}s")
    if CONTROLLER_ENABLED:
        pass  # replace later with real controller command
    time.sleep(hold + after)


def mash(buttons, duration=1.0, min_gap=0.06, max_gap=0.18):
    print(f"[ACTION] MASH {buttons} for {duration:.2f}s")
    end_time = time.time() + duration
    while time.time() < end_time:
        press(random.choice(buttons), hold=jitter(0.04, 0.10), after=jitter(min_gap, max_gap))


def soft_reset():
    print("[ACTION] SOFT RESET (A+B+X+Y placeholder)")
    if CONTROLLER_ENABLED:
        pass
    time.sleep(1.0)


# ============================================================
# State machine
# ============================================================

def detect_state(frame):
    on_info, info_scores = is_pokemon_info_page(frame)

    if on_info:
        return "POKEMON_INFO", info_scores
    if is_summary_screen(frame):
        return "SUMMARY_SCREEN", {}
    if is_options_screen(frame):
        return "OPTIONS_SCREEN", {}
    if is_yes_prompt_page(frame):
        return "YES_PROMPT", {}
    if is_continue_page(frame):
        return "CONTINUE_PAGE", {}
    if is_press_start_page(frame):
        return "PRESS_START_PAGE", {}
    return "OTHER", {}


def overlay_state(frame, state, extra_lines=None, color=(0, 255, 255)):
    display = frame.copy()
    cv2.putText(display, f"STATE: {state}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if extra_lines:
        y = 80
        for line in extra_lines:
            cv2.putText(display, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 30
    return display


def main():
    cap = find_capture_device()
    if cap is None:
        print("No working capture device found.")
        return

    print("Running hunt loop in MONITOR / PLACEHOLDER mode.")
    print("This means the terminal actions are only suggestions until controller integration is added.")
    print("Press q to quit.")

    attempt = 1
    phase = "RESET_AND_FIND_PRESS_START"
    phase_announced = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            state, debug = detect_state(frame)
            lines = [f"attempt={attempt}", f"phase={phase}", f"detected={state}"]

            if state == "POKEMON_INFO":
                shiny = analyze_shiny(frame)
                lines.extend([
                    f"yellow={shiny['yellow_score']:.4f}",
                    f"cyan={shiny['cyan_score']:.4f}",
                    f"purple={shiny['purple_score']:.4f}",
                    f"shiny={shiny['is_shiny']}",
                ])
                color = (0, 0, 255) if shiny["is_shiny"] else (0, 255, 0)
            else:
                color = (0, 255, 255)

            display = overlay_state(frame, state, lines, color)
            cv2.imshow("Hunt Loop Preview", display)

            if phase != phase_announced:
                print(f"\n[PHASE] {phase}")
                phase_announced = phase

            # ------------------------------------------------
            # Phase logic aligned to your PDF
            # ------------------------------------------------

            if phase == "RESET_AND_FIND_PRESS_START":
                if state == "PRESS_START_PAGE":
                    print("[STATE HIT] PRESS_START_PAGE")
                    phase = "START_GAME"
                else:
                    print("[PLAN] mash Y/X/B/A until PRESS_START_PAGE")
                    mash(["y", "x", "b", "a"], duration=0.8)

            elif phase == "START_GAME":
                if state == "CONTINUE_PAGE":
                    print("[STATE HIT] CONTINUE_PAGE")
                    phase = "CONTINUE_AND_MASH_A"
                else:
                    print("[PLAN] mash +/X/A until CONTINUE_PAGE")
                    mash(["+", "x", "a"], duration=0.8)

            elif phase == "CONTINUE_AND_MASH_A":
                if state == "YES_PROMPT":
                    print("[STATE HIT] YES_PROMPT (claim Charmander)")
                    phase = "CLAIM_AND_PROGRESS_TO_NICKNAME"
                else:
                    print("[PLAN] press A once, then mash A until YES_PROMPT")
                    press("a", hold=0.08, after=0.20)
                    mash(["a"], duration=1.0)

            elif phase == "CLAIM_AND_PROGRESS_TO_NICKNAME":
                # PDF says at 18_so: press A, then A once more, then mash B until nickname
                print("[PLAN] press A, press A, then mash B until nickname page")
                press("a", hold=0.08, after=0.20)
                press("a", hold=0.08, after=0.20)
                mash(["b"], duration=1.0)
                if state == "YES_PROMPT":
                    # nickname page also looks like yes/no prompt, so once we see prompt again after B-mash,
                    # we treat it as the nickname prompt stage
                    phase = "DECLINE_NICKNAME"

            elif phase == "DECLINE_NICKNAME":
                # PDF says at nickname: press B, then keep randomly pressing B until 23_wait_4_command
                if state == "OPTIONS_SCREEN":
                    print("[STATE HIT] OPTIONS_SCREEN")
                    phase = "OPEN_PARTY"
                else:
                    print("[PLAN] press B, then mash B until free-command state, then +")
                    press("b", hold=0.08, after=0.20)
                    mash(["b"], duration=0.8)
                    press("+", hold=0.08, after=0.30)

            elif phase == "OPEN_PARTY":
                if state == "SUMMARY_SCREEN":
                    print("[STATE HIT] SUMMARY_SCREEN")
                    phase = "OPEN_INFO_PAGE"
                else:
                    print("[PLAN] press A to Pokemon, then A to Charmander")
                    press("a", hold=0.08, after=0.25)
                    press("a", hold=0.08, after=0.25)

            elif phase == "OPEN_INFO_PAGE":
                if state == "POKEMON_INFO":
                    print("[STATE HIT] POKEMON_INFO")
                    phase = "CHECK_SHINY"
                else:
                    print("[PLAN] press A to Summary / Pokemon Info")
                    press("a", hold=0.08, after=0.30)

            elif phase == "CHECK_SHINY":
                shiny = analyze_shiny(frame)
                print(f"[SHINY CHECK] yellow={shiny['yellow_score']:.6f} cyan={shiny['cyan_score']:.6f} purple={shiny['purple_score']:.6f}")
                print(f"[SHINY CHECK] star={shiny['star_detected']} border={shiny['border_detected']} final={shiny['is_shiny']}")
                if shiny["is_shiny"]:
                    print(f"\n***** SHINY FOUND ON ATTEMPT {attempt} *****")
                    print("[PLAN] keep alive by oscillating left/right")
                    # placeholder keep-alive
                    press("left", hold=0.08, after=0.25)
                    press("right", hold=0.08, after=0.25)
                else:
                    print("[RESULT] not shiny -> soft reset and restart loop")
                    soft_reset()
                    attempt += 1
                    phase = "RESET_AND_FIND_PRESS_START"

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()