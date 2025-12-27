import cv2
import numpy as np
import time
import glob
import os
from google import genai
from google.genai import types
import base64
from openai import OpenAI
import logging
from switchbot_direction_controller import SwitchBotAuth, SwitchBotClient, DirectionSwitchController
from dotenv import load_dotenv
from pathlib import Path
from langtrace_python_sdk import langtrace
import re


#Load langtrace and logger details
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
langtrace.init(api_key = os.getenv("LANGTRACE_API_KEY"))
logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.getenv("LOGGER_NAME"), encoding='utf-8', level=logging.DEBUG, format='%(levelname)s:%(message)s')

# ------------ CONFIG ------------
SAVE_INTERVAL = 10             # seconds between minimap saves
CAPTURE_DIR = os.getenv('CAPTURE_DIRECTORY')
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Global state for minimap ROI (in webcam coordinates)
mm_roi = None             # (x1, y1, x2, y2)
dragging = False
drag_start = None
drag_end = None
last_save_time = 0.0


def get_latest_image(folder_name):
    list_of_files = glob.glob(folder_name) 
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def extract_direction(text):
    # Look for whole words matching allowed directions
    match = re.search(r"\b(UP|LEFT|RIGHT)\b", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None

def directions_executor(direction):
    auth = SwitchBotAuth.from_env()
    client = SwitchBotClient(auth)
    controller = DirectionSwitchController(client, DirectionSwitchController.mapping_from_env())
    controller.trigger(direction)



def call_llm(folder_name, llm_value, token, prompt):
    # Load latest image
    image_path = get_latest_image(f"{folder_name}*")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # -------------------------
    # Gemini 2.5 Flash
    # -------------------------
    if llm_value == "gemini-2.5-flash":
        client = genai.Client(api_key=token)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                prompt,
            ],
        )

        return response.text.strip()

    # -------------------------
    # OpenAI GPT-5-nano (Vision)
    # -------------------------
    elif llm_value == "gpt-5-nano":
        print(prompt)
        client = OpenAI(api_key=token)

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{image_b64}"

        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )
        print(response.output_text)
        return response.output_text

    else:
        raise ValueError(f"Unsupported llm_value: {llm_value}")
    

# ------------ MOUSE CALLBACK ------------

def webcam_mouse_cb(event, x, y,  flags, param):
    """
    Mouse callback for the Webcam window:
    left-click and drag a rectangle around the minimap  or the TV ONCE.
    """
    global dragging, drag_start, drag_end, mm_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        drag_start = (x, y)
        drag_end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        drag_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        drag_end = (x, y)

        x1, y1 = drag_start
        x2, y2 = drag_end
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        mm_roi = (x1, y1, x2, y2)

        print("[MM] Minimap or TV ROI set (webcam pixels):", mm_roi)
        print("[MM] You can now let it run; it will crop & save from this region.")


# ------------ MAIN LOOP ------------

def main():
    global mm_roi, dragging, drag_start, drag_end, last_save_time

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

 # Try to force higher capture resolution:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cv2.namedWindow("Webcam")
    cv2.setMouseCallback("Webcam", webcam_mouse_cb)

    print("Instructions:")
    print("  1. Point laptop so the whole TV is visible.")
    print("  2. In the 'Webcam' window, LEFT-CLICK and drag a box around the minimap or the entire TV screen once.")
    print("  3. After releasing the mouse, the minimap ROI is locked.")
    print("  4. Script will crop that region each frame, save every X seconds",
          SAVE_INTERVAL, "seconds to", CAPTURE_DIR, "and run the direction stub.")
    print("  5. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame from camera; exiting.")
            break

        # Draw current drag rectangle, if dragging
        if dragging and drag_start and drag_end:
            x1, y1 = drag_start
            x2, y2 = drag_end
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 255), 2)

        minimap = None

        if mm_roi is not None:
            x1, y1, x2, y2 = mm_roi

            # clamp to frame size
            h, w = frame.shape[:2]
            x1_c = max(0, min(w - 1, x1))
            x2_c = max(0, min(w,     x2))
            y1_c = max(0, min(h - 1, y1))
            y2_c = max(0, min(h,     y2))

            minimap = frame[y1_c:y2_c, x1_c:x2_c]

    
            cv2.rectangle(frame, (x1_c, y1_c), (x2_c, y2_c),
                          (0, 0, 255), 2)
            blur = cv2.GaussianBlur(minimap, (0, 0), sigmaX=1.2)
            sharpened = cv2.addWeighted(minimap, 1.6, blur, -0.6, 0)

            '''# --- Denoise ---
                den = cv2.fastNlMeansDenoisingColored(minimap, None, 10, 10, 7, 21)

                # --- Upscale ---
                up = cv2.resize(den, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

                # --- Sharpen ---
                blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.4)
                sharp = cv2.addWeighted(up, 1.7, blur, -0.7, 0)

                # Use this as your final minimap
                final_minimap = sharp
            '''

            # Save minimap every SAVE_INTERVAL seconds
            now = time.time()
            if now - last_save_time >= SAVE_INTERVAL:
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(CAPTURE_DIR,
                                        f"minimap_{ts}.png")
                cv2.imwrite(out_path, sharpened)
                print(f"[SAVE] {out_path}")
                last_save_time = now


            # Direction stub
            #direction = call_llm(CAPTURE_DIR,'gemini-2.5-flash',os.getenv("GEMINI_TOKEN"), os.getenv("OPEN_AI_PROMPT"))
            direction = call_llm(CAPTURE_DIR,'gpt-5-nano',os.getenv("OPEN_AI_TOKEN"), os.getenv("OPEN_AI_PROMPT"))
            direction_reasoning = (direction)
            direction = direction_reasoning.split(':')[0]
            reasoning = direction_reasoning.split(':')[1]
            logger.info(direction_reasoning)
            direction = extract_direction(direction)
            directions_executor(direction)


            cv2.putText(frame, f"Dir: {direction}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 2, cv2.LINE_AA)


            # Show minimap in its own window too
            cv2.imshow("Minimap", sharpened)

        # Show main webcam view
        if mm_roi is None:
            cv2.putText(frame, "Drag a box around the minimap or TV",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1)
        if key != -1:
            key = key & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
