 from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
from skimage.color import rgb2lab
import mediapipe as mp
from datetime import datetime

app = Flask(__name__)

def genera_maschera_frontale(image_rgb):
    h, w = image_rgb.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    results = face_mesh.process(image_rgb)
    mask = np.zeros((h, w), dtype=np.uint8)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        punti_viso = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                      365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,
                      58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        points = np.array([[int(l.x * w), int(l.y * h)] for i, l in enumerate(landmarks) if i in punti_viso])
        if len(points) > 0:
            cv2.fillPoly(mask, [points], 255)
    return mask > 0

@app.route("/process", methods=["POST"])
def process_image():
    if 'file' not in request.files or 'dob' not in request.form:
        return jsonify({"error": "Serve immagine e data di nascita (dob)"}), 400

    file = request.files['file']
    dob_str = request.form['dob']
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        today = datetime.today()
        eta = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception as e:
        return jsonify({"error": f"Data non valida: {str(e)}"}), 400

    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    mask = genera_maschera_frontale(image_rgb)
    lab = rgb2lab(image_np)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    L_mean = np.mean(L[mask])

    def stima_fototipo(L_val):
        if L_val > 80: return "Tipo I (molto chiara)"
        elif L_val > 70: return "Tipo II (chiara)"
        elif L_val > 60: return "Tipo III (medio-chiara)"
        elif L_val > 50: return "Tipo IV (olivastra)"
        elif L_val > 40: return "Tipo V (marrone)"
        else: return "Tipo VI (molto scura)"
    fototipo = stima_fototipo(L_mean)

    return jsonify({
        "età": eta,
        "fototipo": fototipo,
        "L* medio": round(L_mean, 1)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
