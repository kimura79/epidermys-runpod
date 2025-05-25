import base64
import json
import io
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2lab
import mediapipe as mp
from datetime import datetime
import sys

def genera_maschera_frontale(image_rgb):
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            punti_viso = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                          365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,
                          58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            points = np.array([[int(l.x * w), int(l.y * h)] for i, l in enumerate(landmarks) if i in punti_viso])
            if len(points) > 0:
                cv2.fillPoly(mask, [points], 255)
    return mask > 0

def stima_fototipo(L_val):
    if L_val > 80: return "Tipo I (molto chiara)"
    elif L_val > 70: return "Tipo II (chiara)"
    elif L_val > 60: return "Tipo III (medio-chiara)"
    elif L_val > 50: return "Tipo IV (olivastra)"
    elif L_val > 40: return "Tipo V (marrone)"
    else: return "Tipo VI (molto scura)"

def handler(event):
    try:
        img_b64 = event["input"]["image_base64"]
        dob_str = event["input"]["data_nascita"]

        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        today = datetime.today()
        eta = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

        image_bytes = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        mask = genera_maschera_frontale(image_rgb)
        lab = rgb2lab(image_np)
        L_mean = np.mean(lab[:, :, 0][mask])
        fototipo = stima_fototipo(L_mean)

        return {
            "output": {
                "età": eta,
                "fototipo": fototipo,
                "L* medio": round(L_mean, 1)
            }
        }

    except Exception as e:
        return { "error": str(e) }

if __name__ == "__main__":
    try:
        print("Awaiting input JSON from stdin...")
        event = json.load(sys.stdin)
        output = handler(event)
        print(json.dumps(output))
    except Exception as e:
        print(json.dumps({ "error": str(e) }))
