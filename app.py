import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from skimage.color import rgb2lab
import io

def genera_maschera_frontale(image_rgb):
    # Codice della maschera integrato qui (semplificato)
    h, w = image_rgb.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    return mask  # Maschera fittizia (sostituire con logica reale)

def run(image, data_nascita=None):
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    mask = genera_maschera_frontale(image_rgb)
    lab = rgb2lab(image_np)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    L_mask = np.mean(L[mask])

    def stima_fototipo(L_val):
        if L_val > 80: return "Tipo I"
        elif L_val > 70: return "Tipo II"
        elif L_val > 60: return "Tipo III"
        elif L_val > 50: return "Tipo IV"
        elif L_val > 40: return "Tipo V"
        else: return "Tipo VI"
    
    fototipo = stima_fototipo(L_mask)

    # Stima posa
    posa = "non rilevata"
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_np)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            h, w = image_np.shape[:2]
            naso = np.array([lm[1].x * w, lm[1].y * h])
            mento = np.array([lm[152].x * w, lm[152].y * h])
            angolo = np.degrees(np.arctan2(mento[0] - naso[0], mento[1] - naso[1]))
            if abs(angolo) < 15: posa = "frontale"
            elif angolo > 40: posa = "profilo destro"
            elif angolo < -40: posa = "profilo sinistro"
            elif angolo > 15: posa = "3/4 destro"
            elif angolo < -15: posa = "3/4 sinistro"

    # Età da data di nascita se presente
    from datetime import datetime
    eta = None
    if data_nascita:
        oggi = datetime.today()
        try:
            nascita = datetime.strptime(data_nascita, "%Y-%m-%d")
            eta = oggi.year - nascita.year - ((oggi.month, oggi.day) < (nascita.month, nascita.day))
        except:
            eta = "Formato data errato"

    return {
        "fototipo": fototipo,
        "posa": posa,
        "L*": round(L_mask, 1),
        "età": eta
    }
