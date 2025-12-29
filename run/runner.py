import cv2
import numpy as np
import time
import glob
import os
import base64
import logging
import re

from switchbot_direction_controller import SwitchBotAuth, SwitchBotClient, DirectionSwitchController
from dotenv import load_dotenv
from pathlib import Path
from langtrace_python_sdk import langtrace, inject_additional_attributes
from opentelemetry import trace

from openai import OpenAI
from google import genai
from google.genai import types
import anthropic              # Claude
from groq import Groq         # Llama via Groq

# ------------ CONFIG & SETUP ------------
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
langtrace.init(api_key=os.getenv("LANGTRACE_API_KEY"), write_spans_to_console=False)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.getenv("LOGGER_NAME"),
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(levelname)s:%(message)s",
)

SAVE_INTERVAL = 10
CAPTURE_DIR = os.getenv("CAPTURE_DIRECTORY")
os.makedirs(CAPTURE_DIR, exist_ok=True)

# minimap ROI (webcam coordinates)
mm_roi = None
dragging = False
drag_start = None
drag_end = None
last_save_time = 0.0

# -----------------------------------------------------------
#                     HELPER FUNCTIONS
# -----------------------------------------------------------

def get_latest_image(folder_name):
    files = glob.glob(folder_name)
    return max(files, key=os.path.getctime)

def extract_direction(text):
    m = re.search(r"\b(UP|LEFT|RIGHT)\b", text, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def directions_executor(direction):
    auth = SwitchBotAuth.from_env()
    client = SwitchBotClient(auth)
    controller = DirectionSwitchController(client, DirectionSwitchController.mapping_from_env())
    controller.trigger(direction)


# -----------------------------------------------------------
#                     LLM DISPATCHER
# -----------------------------------------------------------

def call_llm(folder_name, llm_value, token, prompt):
    """
    folder_name: directory containing images; e.g. CAPTURE_DIR
    llm_value:   "gemini-2.5-flash" / "gpt-5-nano" / "claude-sonnet-4-5" / "llama-4-scout"
    token:       API key
    prompt:      navigation prompt expecting "DIRECTION:REASONING"
    """
    # --- load latest minimap image ---
    image_path = get_latest_image(f"{folder_name}*")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    PNG_MIME = "image/png"  # always PNG

    ##OpenTelemetryEvent to pass custom attributes
    tracer = trace.get_tracer("llm_agent_run")
    with tracer.start_as_current_span(llm_value) as span:
        #set attributes AFTER the span object exists
        span.set_attribute("ctx.run_id", "1")
        span.set_attribute("ctx.decision_source", "responses-test")

    # ------------------------- Gemini -------------------------
    if llm_value == "gemini-2.5-flash":
        client = genai.Client(api_key=token)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=PNG_MIME),
                prompt,
            ],
        )
        span.set_attribute("ctx.next_action", resp.text.strip())
        return resp.text.strip()

    # ------------------------- GPT-5 Nano Vision -------------------------
    elif llm_value == "gpt-5-nano":
        client = OpenAI(api_key=token)
        data_url = f"data:{PNG_MIME};base64,{image_b64}"
        resp = client.responses.create(
            model="gpt-5-nano",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
        )
        span.set_attribute("ctx.next_action", resp.output_text)
        return resp.output_text

    # ------------------------- Claude -------------------------
    elif llm_value == "claude-sonnet-4-5":
        client = anthropic.Anthropic(api_key=token)
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": PNG_MIME, "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text_blocks = [b.text for b in resp.content if b.type == "text"]
        span.set_attribute("ctx.next_action", "".join(text_blocks).strip())
        return "".join(text_blocks).strip()

    # ------------------------- Llama -------------------------
    elif llm_value == "llama-4-scout":
        client = Groq(api_key=token)
        data_url = f"data:{PNG_MIME};base64,{image_b64}"
        out = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_completion_tokens=1024,
            stream=False,
        )
        span.set_attribute("ctx.next_action", out.choices[0].message.content.strip())
        return out.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unsupported llm_value: {llm_value}")


# -----------------------------------------------------------
#                     MOUSE CALLBACK
# -----------------------------------------------------------

def webcam_mouse_cb(event, x, y, flags, param):
    global dragging, drag_start, drag_end, mm_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True; drag_start = (x, y); drag_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        drag_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False; drag_end = (x, y)
        (x1, y1), (x2, y2) = sorted([drag_start, drag_end])
        mm_roi = (x1, y1, x2, y2)
        print("[MM] ROI set:", mm_roi)


# -----------------------------------------------------------
#                        MAIN LOOP
# -----------------------------------------------------------

def main():
    global mm_roi, dragging, drag_start, drag_end, last_save_time

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("Could not open webcam.")

    cv2.namedWindow("Webcam")
    cv2.setMouseCallback("Webcam", webcam_mouse_cb)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame; exiting."); break

        if dragging and drag_start and drag_end:
            cv2.rectangle(frame, drag_start, drag_end, (0,255,255), 2)

        # ---- after ROI selected ----
        if mm_roi:
            x1,y1,x2,y2 = mm_roi
            h,w = frame.shape[:2]
            x1,x2 = max(0,x1), min(w,x2)
            y1,y2 = max(0,y1), min(h,y2)
            crop = frame[y1:y2, x1:x2]

            blur = cv2.GaussianBlur(crop,(0,0),1.2)
            sharp = cv2.addWeighted(crop,1.6,blur,-0.6,0)

            # save PNG every interval
            if time.time() - last_save_time >= SAVE_INTERVAL:
                ts = time.strftime("%Y%m%d_%H%M%S")
                out = os.path.join(CAPTURE_DIR, f"minimap_{ts}.png")
                cv2.imwrite(out, sharp)
                last_save_time = time.time()
                print("[SAVE]", out)

            # ---- LLM SELECTION (change here to swap models) ----
            #direction_text = call_llm(
                    #CAPTURE_DIR,
                    # "gpt-5-nano", os.getenv("OPEN_AI_TOKEN"), os.getenv("OPEN_AI_PROMPT")
                    # "gemini-2.5-flash", os.getenv("GEMINI_TOKEN"), os.getenv("OPEN_AI_PROMPT")
                    # "claude-sonnet-4-5", os.getenv("CLAUDE_API_KEY"), os.getenv("CLAUDE_PROMPT")
                   # "llama-4-scout", os.getenv("GROQ_API_KEY"), os.getenv("LLAMA_PROMPT")
                #)
            LLM_ROTATION = [
                ("gpt-5-nano",       os.getenv("OPEN_AI_TOKEN"),   os.getenv("OPEN_AI_PROMPT"))
                #,
               # ("claude-sonnet-4-5",os.getenv("CLAUDE_API_KEY"),  os.getenv("CLAUDE_PROMPT")),
                #("llama-4-scout",    os.getenv("GROQ_API_KEY"),    os.getenv("LLAMA_PROMPT")),
            ]

            for llm_name, key, prompt in LLM_ROTATION:
                print(f"\n🔍 Testing: {llm_name}")
                direction_text = call_llm(CAPTURE_DIR, llm_name, key, prompt)

                # Parse response
                parts = direction_text.split(":", 1)
                direction_raw = parts[0] if parts else ""
                reasoning = parts[1] if len(parts)>1 else ""

                direction = extract_direction(direction_raw) or "UNKNOWN"
                logger.info(f"{llm_name} → {direction_text}")

                print(f"🧭 {llm_name} → Direction: {direction}")
                print(f"💡 Reasoning: {reasoning}")
            '''
            # Parse "DIRECTION:REASONING"
            parts = direction_text.split(":",1)
            direction_raw = parts[0] if parts else ""
            reasoning = parts[1] if len(parts)>1 else ""
            direction = extract_direction(direction_raw)
           #if direction: directions_executor(direction)
           '''

            logger.info(direction_text)
            cv2.putText(frame, f"Dir: {direction or 'UNKNOWN'}",(10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)

            cv2.imshow("Minimap", sharp)

        else:
            cv2.putText(frame,"Drag ROI around minimap or TV",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
