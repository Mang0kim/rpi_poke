#!/usr/bin/env python3
# -------------------------------
# 1) Import required libraries
# -------------------------------
import cv2
import sys

# -------------------------------
# 2) Global settings / variables
# -------------------------------
VIDEO_FILES = ["01.mp4", "02.mp4", "03.mp4"]
WINDOW_NAME = "Player"
HEADLESS = False   # Set True for headless mode (no display), False for fullscreen display

# -------------------------------
# 3) Functions
# -------------------------------
def play_video(path: str) -> bool:
    """Play a single video file once. Return False if open/read fails, True otherwise."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {path}")
        return False

    if not HEADLESS:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # Fullscreen mode
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of file
        if not HEADLESS:
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Esc key
                cap.release()
                return True
        else:
            # In headless mode, just read frames without display
            continue

    cap.release()
    return True

def play_sequence_once(files) -> None:
    """Play 01 -> 02 -> 03 exactly once each."""
    for f in files:
        ok = play_video(f)
        if not ok:
            continue

def cleanup():
    """Release all windows."""
    if not HEADLESS:
        cv2.destroyAllWindows()

# -------------------------------
# 4) Main loop
# -------------------------------
if __name__ == "__main__":
    try:
        while True:
            play_sequence_once(VIDEO_FILES)
            break  # play once then exit (remove this 'break' for endless loop)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
        sys.exit(0)
