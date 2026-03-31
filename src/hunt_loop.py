"to run again from your project root, use this command: ./.venv/bin/python src/hunt_loop.py"
import cv2
import numpy as np
import random
import time


# ============================================================
# Vision helpers
# ============================================================

def crop_relative(image, x_min, x_max, y_min, y_max):
    h, w = image.shape[:2]
    return image[int(h * y_min):int(h * y_max), int(w * x_min):int(w * x_max)]

def mask_ratio(mask):
    return float(cv2.countNonZero(mask)) / float(mask.shape[0] * mask.shape[1])

def hsv_mask(image, lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

def yellow_mask_bgr(img):    return hsv_mask(img, [15,  80,  80],  [40,  255, 255])
def cyan_mask_bgr(img):      return hsv_mask(img, [75,  50,  50],  [110, 255, 255])
def purple_mask_bgr(img):    return hsv_mask(img, [120, 40,  40],  [165, 255, 255])
def orange_panel_mask(img):  return hsv_mask(img, [5,   20,  150], [25,  120, 255])
def blue_header_mask(img):   return hsv_mask(img, [95,  80,  80],  [120, 255, 255])
def beige_header_mask(img):  return hsv_mask(img, [15,  20,  150], [40,  120, 255])


# ============================================================
# Page detectors
#
# is_pokemon_info_page and analyze_shiny are fully validated.
# All other detectors are written from visual inspection of the
# PDF screenshots and NEED VALIDATION against real Switch captures.
# To validate: run capture_test.py, save screenshots at each screen,
# then add score-printing tests in star_detect_test.py and tune
# the thresholds below until each detector fires cleanly.
# ============================================================

def is_pokemon_info_page(image):
    """Validated. Gate detector for screen 27_pokemon_info."""
    left  = crop_relative(image, 0.02, 0.50, 0.08, 0.62)
    right = crop_relative(image, 0.52, 0.98, 0.10, 0.72)
    hdr   = crop_relative(image, 0.00, 1.00, 0.00, 0.12)
    p = mask_ratio(purple_mask_bgr(left))
    o = mask_ratio(orange_panel_mask(right))
    b = mask_ratio(blue_header_mask(hdr))
    e = mask_ratio(beige_header_mask(hdr))
    on_page = (p > 0.08) and (o > 0.10) and (b > 0.08) and (e > 0.05)
    return on_page, {"purple": p, "orange": o, "blue": b, "beige": e}

def analyze_shiny(image):
    """Validated. Only call when is_pokemon_info_page fires."""
    star   = crop_relative(image, 0.405, 0.475, 0.205, 0.295)
    border = crop_relative(image, 0.02,  0.435, 0.08,  0.60)
    y = mask_ratio(yellow_mask_bgr(star))
    c = mask_ratio(cyan_mask_bgr(border))
    p = mask_ratio(purple_mask_bgr(border))
    star_det   = y > 0.010
    border_det = (c > 0.080) and (p < 0.050)
    return {
        "is_shiny":        star_det or border_det,
        "star_detected":   star_det,
        "border_detected": border_det,
        "yellow_score": y, "cyan_score": c, "purple_score": p,
    }

def is_press_start_page(image):
    """
    Screen 11: Pokemon LeafGreen title screen.
    Signature: green bar top, green bar bottom, orange/salmon center.
    NEEDS VALIDATION — tune g_top/g_bot/o_ctr thresholds with real screenshots.
    """
    top    = crop_relative(image, 0.00, 1.00, 0.00, 0.12)
    bottom = crop_relative(image, 0.00, 1.00, 0.88, 1.00)
    center = crop_relative(image, 0.05, 0.95, 0.12, 0.80)
    g_top = mask_ratio(hsv_mask(top,    [40, 80, 80], [85, 255, 255]))
    g_bot = mask_ratio(hsv_mask(bottom, [40, 80, 80], [85, 255, 255]))
    o_ctr = mask_ratio(hsv_mask(center, [5,  80, 150], [20, 255, 255]))
    return (g_top > 0.15) and (g_bot > 0.15) and (o_ctr > 0.10)

def is_continue_page(image):
    """
    Screen 12: CONTINUE / NEW GAME dialog.
    Unique signature: the periwinkle/purple background fills the entire frame,
    so blue in the large dialog crop is extremely high (0.32) compared to all
    game-scene screens where blue is only a small box border (< 0.12).
    Validated scores — 12_continue: white=0.47 blue=0.32 → True
                       all others:  blue < 0.12           → False
    """
    dialog = crop_relative(image, 0.06, 0.88, 0.05, 0.88)
    white  = mask_ratio(hsv_mask(dialog, [0,   0,  200], [180, 40,  255]))
    blue   = mask_ratio(hsv_mask(dialog, [95,  40,  80], [125, 255, 255]))
    return (white > 0.30) and (blue > 0.20)

def is_yes_prompt_page(image):
    """
    Screens 18 (claim Charmander) and 21 (nickname): YES/NO box bottom-right.
    Both show an identical small YES/NO box. Phase context distinguishes them.
    Crop targets only the YES/NO box position (y:28-58%, x:62-85%).
    The options menu does NOT extend into this y-range from the top.
    Validated scores — 18_so: 0.143, 21_nickname: 0.104 → True
                       24_options: 0.045, 11_press_start: 0.006 → False
    (12_continue and 25/26 are caught earlier in detect_state priority)
    """
    crop = crop_relative(image, 0.62, 0.85, 0.28, 0.58)
    return mask_ratio(hsv_mask(crop, [95, 60, 100], [125, 255, 255])) > 0.08

def is_options_screen(image):
    """
    Screen 24: POKEMON/BAG/MANNY/SAVE/OPTION/EXIT menu box on right side.
    Unique signature: the menu border reaches the very TOP of the screen (y=0),
    while the YES/NO box only starts at y~28%. Crop the top-right corner
    specifically to exploit this difference.
    Validated scores — 24_options: 0.152 → True
                       18_so: 0.003, 21_nickname: 0.003 → False
    (12_continue scores 0.335 here but is caught by is_continue_page first)
    """
    top_right = crop_relative(image, 0.62, 0.92, 0.00, 0.15)
    return mask_ratio(hsv_mask(top_right, [95, 60, 100], [125, 255, 255])) > 0.10

def is_summary_screen(image):
    """
    Screens 25 (party) and 26 (SUMMARY/ITEM/CANCEL popup).
    Signature: strong teal background dominates the left half of screen.
    Validated scores — 25_party: 0.663, 26_summary: 0.650 → True
                       all others: < 0.08                  → False
    """
    left = crop_relative(image, 0.00, 0.65, 0.10, 0.90)
    return mask_ratio(hsv_mask(left, [75, 40, 80], [110, 255, 255])) > 0.10

def detect_state(frame):
    """
    Priority order is critical — each detector assumes the earlier ones
    have already been ruled out.

    Key ordering decisions:
    - CONTINUE is checked BEFORE OPTIONS and YES_PROMPT because the
      periwinkle background of 12_continue causes those detectors to
      also fire on it. Catching continue early prevents false positives.
    - PRESS_START is checked early because it has zero overlap with anything.
    - OPTIONS before YES_PROMPT because 24_options has some blue in the
      yes_prompt crop region (score 0.045), but options fires first.

    Validated: all 7 real screenshots route to exactly the correct label.
    """
    on_info, scores = is_pokemon_info_page(frame)
    if on_info:                    return "POKEMON_INFO",     scores
    if is_summary_screen(frame):   return "SUMMARY_SCREEN",   {}
    if is_press_start_page(frame): return "PRESS_START_PAGE", {}
    if is_continue_page(frame):    return "CONTINUE_PAGE",    {}
    if is_options_screen(frame):   return "OPTIONS_SCREEN",   {}
    if is_yes_prompt_page(frame):  return "YES_PROMPT",       {}
    return "OTHER", {}


# ============================================================
# Capture
# ============================================================

def find_capture_device(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened(): cap.release(); continue
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Using capture device index {i}")
            return cap
        cap.release()
    return None


# ============================================================
# Controller placeholder layer
# When the controller is wired (Arduino/Pico via serial),
# replace the `pass` lines with real serial commands.
# ============================================================

CONTROLLER_ENABLED = False

def press(button, hold=0.06, after=0.0):
    """
    Fire a single button press.
    hold  = how long the button is held down (seconds)
    after = pause after release before returning (seconds)
    Keep hold + after small here; ACTION_COOLDOWN in the loop
    controls the real gap between successive button pulses.
    """
    if CONTROLLER_ENABLED:
        pass  # TODO: send serial command, e.g. ser.write(f"PRESS:{button}\n".encode())
    time.sleep(hold)
    if after > 0:
        time.sleep(after)

def soft_reset():
    """Simultaneous A+B+X+Y press to soft-reset the game."""
    print("[ACTION] SOFT RESET (A+B+X+Y)")
    if CONTROLLER_ENABLED:
        pass  # TODO: send simultaneous button press command
    time.sleep(0.5)


# ============================================================
# Overlay
# ============================================================

def draw_overlay(frame, raw_state, confirmed, phase, attempt, extra_lines=None):
    d = frame.copy()
    cv2.putText(d, f"STATE: {raw_state}",  (20, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(d, f"CONF:  {confirmed}",  (20, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)
    cv2.putText(d, f"PHASE: {phase}",      (20, 95),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    cv2.putText(d, f"attempt: {attempt}",  (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    if extra_lines:
        y = 160
        for line in extra_lines:
            cv2.putText(d, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 28
    return d


# ============================================================
# Main hunt loop
# ============================================================

def main():
    cap = find_capture_device()
    if cap is None:
        print("No capture device found.")
        return

    print("Hunt loop running.")
    print(f"CONTROLLER_ENABLED = {CONTROLLER_ENABLED}")
    print("Button presses print to terminal. Real hardware fires when CONTROLLER_ENABLED = True.")
    print("Press q to quit.\n")

    # ----------------------------------------------------------------
    # Loop-level state
    # ----------------------------------------------------------------
    attempt          = 1
    phase            = "RESET_AND_FIND_PRESS_START"
    phase_changed_at = time.time()

    # How long to wait after a phase change before firing actions.
    # This gives the game time to render the next screen before we act.
    PHASE_DWELL = 0.8  # seconds

    # ----------------------------------------------------------------
    # State confirmation streak
    # A state must appear in CONFIRM_FRAMES consecutive frames before
    # it is treated as "confirmed" and used to drive phase transitions.
    # This prevents transient frames or partial screen renders from
    # triggering premature phase changes.
    # ----------------------------------------------------------------
    streak_state = None
    streak_count = 0
    CONFIRM_FRAMES = 4

    # ----------------------------------------------------------------
    # Action throttle
    # The frame loop runs as fast as the capture card allows (~30-60 fps).
    # ACTION_COOLDOWN limits how often we actually send button presses,
    # keeping the loop non-blocking while still mashing at a reasonable rate.
    # ----------------------------------------------------------------
    last_action_time = 0.0
    ACTION_COOLDOWN  = 0.18  # seconds between button pulses

    # ----------------------------------------------------------------
    # Per-phase sub-state variables
    # All reset by change_phase() on every phase transition.
    # ----------------------------------------------------------------
    claim_a_presses = 0      # CLAIM_CHARMANDER: how many A's fired so far (need 2)
    claim_left_yes  = False  # CLAIM_CHARMANDER: did we observe a non-YES_PROMPT frame
                             #   after the A presses? Guards against re-detecting the
                             #   claim YES_PROMPT before the nickname one appears.
    decline_start   = 0.0   # DECLINE_NICKNAME: timestamp when mashing began
    plus_fired_at   = 0.0   # DECLINE_NICKNAME: timestamp when + was last pressed
    info_a_presses  = 0     # OPEN_PARTY_AND_INFO: how many A's fired (need 3)
    shiny_logged    = False  # CHECK_SHINY: have we printed the result this round?

    # Seconds of B/A mashing before pressing + in DECLINE_NICKNAME.
    # Must be long enough to clear all post-nickname dialogs
    # (22_george_1, 22_george_2) and land at the overworld (23_wait_4_command).
    DECLINE_MASH_DUR = 2.5

    def change_phase(new_phase):
        """Transition to a new phase and reset all sub-state."""
        nonlocal phase, phase_changed_at, streak_state, streak_count
        nonlocal claim_a_presses, claim_left_yes
        nonlocal decline_start, plus_fired_at
        nonlocal info_a_presses, shiny_logged
        print(f"\n[PHASE] {phase} → {new_phase}  (attempt={attempt})")
        phase            = new_phase
        phase_changed_at = time.time()
        streak_state     = None
        streak_count     = 0
        claim_a_presses  = 0
        claim_left_yes   = False
        decline_start    = 0.0
        plus_fired_at    = 0.0
        info_a_presses   = 0
        shiny_logged     = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # ---- detect raw state this frame ----
            raw_state, dbg = detect_state(frame)

            # ---- update confirmation streak ----
            if raw_state == streak_state:
                streak_count += 1
            else:
                streak_state = raw_state
                streak_count = 1
            confirmed = raw_state if streak_count >= CONFIRM_FRAMES else None

            # ---- build shiny overlay lines ----
            extra = []
            if raw_state == "POKEMON_INFO":
                s = analyze_shiny(frame)
                extra = [
                    f"yellow={s['yellow_score']:.4f}  star={'HIT' if s['star_detected'] else 'miss'}",
                    f"cyan={s['cyan_score']:.4f}  border={'HIT' if s['border_detected'] else 'miss'}",
                    f"SHINY = {'*** YES ***' if s['is_shiny'] else 'no'}",
                ]

            cv2.imshow("Hunt Loop",
                       draw_overlay(frame, raw_state, confirmed or "...", phase, attempt, extra))

            # ---- timing gates ----
            now      = time.time()
            in_dwell = (now - phase_changed_at) < PHASE_DWELL
            can_act  = (now - last_action_time) >= ACTION_COOLDOWN

            # ==============================================================
            # Phase state machine
            # Sequence mirrors PDF2 exactly:
            #   1_first_screen → [mash Y/X/B/A] → 11_press_start
            #   11_press_start → [+/X/A]         → 12_continue
            #   12_continue    → [mash A]         → 18_so (claim YES/NO)
            #   18_so          → [A, A, mash B]   → 21_nickname (YES/NO)
            #   21_nickname    → [B, mash B/A, +] → 24_options_screen
            #   24_options     → [A, A, A]         → 27_pokemon_info
            #   27_info        → shiny check        → keep alive OR soft reset
            # ==============================================================

            # ------------------------------------------------------------------
            # Phase 1: RESET_AND_FIND_PRESS_START
            # After a soft reset the game shows copyright/cutscene screens
            # (1–10 in PDF1). Mash Y/X/B/A with random timing to skip them.
            # Random timing is intentional: it seeds the RNG differently each
            # reset, giving a fair probability distribution over shiny outcomes.
            # Transition: confirmed PRESS_START_PAGE detected.
            # ------------------------------------------------------------------
            if phase == "RESET_AND_FIND_PRESS_START":
                if confirmed == "PRESS_START_PAGE":
                    change_phase("START_GAME")
                elif not in_dwell and can_act:
                    btn = random.choice(["y", "x", "b", "a"])
                    press(btn)
                    print(f"[MASH→PRESSSTART] {btn}")
                    last_action_time = now

            # ------------------------------------------------------------------
            # Phase 2: START_GAME
            # At screen 11 (PRESS START), press +, X, or A to advance.
            # Transition: confirmed CONTINUE_PAGE (screen 12).
            # ------------------------------------------------------------------
            elif phase == "START_GAME":
                if confirmed == "CONTINUE_PAGE":
                    change_phase("CONTINUE_AND_MASH_A")
                elif not in_dwell and can_act:
                    btn = random.choice(["+", "x", "a"])
                    press(btn)
                    print(f"[MASH→CONTINUE] {btn}")
                    last_action_time = now

            # ------------------------------------------------------------------
            # Phase 3: CONTINUE_AND_MASH_A
            # At screen 12 (CONTINUE), press A, then keep mashing A.
            # This advances through screens 13 (Previously on quest…),
            # 14, 15 (auto-advances), 16 (overworld/pokeball), 17 (dialog),
            # until the claim YES/NO prompt (18_so) appears.
            # Transition: confirmed YES_PROMPT.
            # ------------------------------------------------------------------
            elif phase == "CONTINUE_AND_MASH_A":
                if confirmed == "YES_PROMPT":
                    change_phase("CLAIM_CHARMANDER")
                elif not in_dwell and can_act:
                    press("a")
                    print("[MASH→CLAIM_PROMPT] a")
                    last_action_time = now

            # ------------------------------------------------------------------
            # Phase 4: CLAIM_CHARMANDER
            # At screen 18_so (claim YES/NO):
            #   - Press A  (confirms YES; cursor is on YES by default)
            #   - Press A  (advances dialog 19_this)
            #   - Mash B   (advances 20_received and arrives at 21_nickname)
            # claim_a_presses tracks the first two mandatory A presses.
            # claim_left_yes ensures we don't confuse the claim YES_PROMPT
            # with the nickname YES_PROMPT: we only transition to
            # DECLINE_NICKNAME once we've seen a non-YES_PROMPT frame
            # after the A presses AND then YES_PROMPT reappears.
            # Transition: confirmed YES_PROMPT after claim_left_yes is True.
            # ------------------------------------------------------------------
            elif phase == "CLAIM_CHARMANDER":
                if claim_a_presses < 2:
                    # Fire the two mandatory A presses; space them slightly
                    if not in_dwell and can_act:
                        press("a", hold=0.08, after=0.10)
                        claim_a_presses += 1
                        print(f"[CLAIM A #{claim_a_presses}/2]")
                        last_action_time = now
                else:
                    # Mark when we've left the claim YES_PROMPT screen
                    if raw_state != "YES_PROMPT":
                        claim_left_yes = True
                    # Only transition when YES_PROMPT re-appears after leaving it
                    if claim_left_yes and confirmed == "YES_PROMPT":
                        change_phase("DECLINE_NICKNAME")
                    elif can_act:
                        press("b")
                        print("[MASH→NICK_PROMPT] b")
                        last_action_time = now

            # ------------------------------------------------------------------
            # Phase 5: DECLINE_NICKNAME
            # At screen 21_nickname (YES/NO for nickname):
            #   - Mash B and A for DECLINE_MASH_DUR seconds.
            #     B declines the nickname (selects NO).
            #     A clears 22_george_1 and 22_george_2 dialogs.
            #     After that we are at 23_wait_4_command (overworld, no UI).
            #   - Then press + once to open the menu (→ screen 24_options_screen).
            #   - Retry + every 1.5s until OPTIONS_SCREEN is confirmed, in case
            #     the overworld wasn't fully loaded when + was first pressed.
            # Transition: confirmed OPTIONS_SCREEN.
            # ------------------------------------------------------------------
            elif phase == "DECLINE_NICKNAME":
                # Capture the timestamp when we first enter this phase
                if decline_start == 0.0:
                    decline_start = now

                if confirmed == "OPTIONS_SCREEN":
                    change_phase("OPEN_PARTY_AND_INFO")
                elif plus_fired_at > 0.0:
                    # + already pressed; retry if OPTIONS_SCREEN hasn't appeared
                    if can_act and (now - plus_fired_at) > 1.5:
                        press("+", hold=0.10)
                        plus_fired_at = now
                        print("[RETRY +] still waiting for options screen")
                        last_action_time = now
                elif (now - decline_start) >= DECLINE_MASH_DUR:
                    # Mash phase complete; fire + to open menu
                    press("+", hold=0.10, after=0.10)
                    plus_fired_at = now
                    print("[ACTION +] opening menu")
                    last_action_time = now
                elif can_act:
                    # Still in mash phase; bias toward b to handle nickname + dialogs
                    btn = random.choice(["b", "b", "a"])
                    press(btn)
                    print(f"[MASH→OVERWORLD] {btn}")
                    last_action_time = now

            # ------------------------------------------------------------------
            # Phase 6: OPEN_PARTY_AND_INFO
            # At screen 24_options_screen, press A three times:
            #   A #1: opens party screen (25_party)
            #   A #2: opens action popup SUMMARY/ITEM/CANCEL (26_summary)
            #   A #3: opens Pokemon Info (27_pokemon_info) — SUMMARY is default
            # Use 0.5s between presses to let each screen fully load.
            # If POKEMON_INFO is confirmed before all 3 A's, transition early.
            # Transition: confirmed POKEMON_INFO.
            # ------------------------------------------------------------------
            elif phase == "OPEN_PARTY_AND_INFO":
                if confirmed == "POKEMON_INFO":
                    change_phase("CHECK_SHINY")
                elif info_a_presses < 3 and not in_dwell:
                    # 0.5s gap between each A press (overrides ACTION_COOLDOWN here)
                    if (now - last_action_time) >= 0.5:
                        press("a", hold=0.08, after=0.05)
                        info_a_presses += 1
                        print(f"[NAVIGATE A #{info_a_presses}/3]")
                        last_action_time = now
                # If all 3 fired and POKEMON_INFO not yet confirmed: just wait

            # ------------------------------------------------------------------
            # Phase 7: CHECK_SHINY
            # On screen 27_pokemon_info, run the shiny detector once and decide.
            # shiny_logged prevents the result from printing every frame.
            # If shiny: transition to KEEP_ALIVE.
            # If not shiny: soft reset and restart from phase 1.
            # If we somehow leave the info page: retry navigation.
            # ------------------------------------------------------------------
            elif phase == "CHECK_SHINY":
                if raw_state == "POKEMON_INFO":
                    s = analyze_shiny(frame)
                    if not shiny_logged:
                        print(f"[SHINY CHECK] attempt={attempt}")
                        print(f"  yellow={s['yellow_score']:.6f}  star={s['star_detected']}")
                        print(f"  cyan={s['cyan_score']:.6f}  border={s['border_detected']}")
                        print(f"  is_shiny={s['is_shiny']}")
                        shiny_logged = True
                    if s["is_shiny"]:
                        print(f"\n{'=' * 52}")
                        print(f"  *** SHINY FOUND on attempt {attempt}! ***")
                        print(f"{'=' * 52}\n")
                        change_phase("KEEP_ALIVE")
                    elif can_act:
                        print(f"[RESULT] not shiny → soft reset (attempt {attempt})")
                        soft_reset()
                        attempt += 1
                        last_action_time = now
                        change_phase("RESET_AND_FIND_PRESS_START")
                else:
                    # Lost the info page; go back and re-navigate
                    if can_act:
                        print("[WARN] left info page unexpectedly → retrying navigation")
                        change_phase("OPEN_PARTY_AND_INFO")

            # ------------------------------------------------------------------
            # Phase 8: KEEP_ALIVE
            # Shiny found. Oscillate left/right so the Switch doesn't sleep
            # until you take over manually.
            # No terminal printing in this phase to avoid log spam.
            # ------------------------------------------------------------------
            elif phase == "KEEP_ALIVE":
                if can_act:
                    btn = random.choice(["left", "right"])
                    press(btn, hold=0.08)
                    last_action_time = now

            # ---- quit ----
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()