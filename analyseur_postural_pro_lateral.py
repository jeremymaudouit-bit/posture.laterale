import io
import math
import os
import tempfile
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from fpdf import FPDF
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Analyseur Postural Latéral", layout="wide")
st.title("🧍 Analyseur Postural Latéral (MediaPipe)")
st.markdown("Mesure sur **un seul côté visible** : jambe, cuisse, tronc et tête par rapport à la verticale.")
st.markdown("---")

mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

pose = load_pose()


def ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    return img


def rotate_if_landscape(img_np_rgb: np.ndarray) -> np.ndarray:
    if img_np_rgb.shape[1] > img_np_rgb.shape[0]:
        img_np_rgb = cv2.rotate(img_np_rgb, cv2.ROTATE_90_CLOCKWISE)
    return img_np_rgb


def to_png_bytes(img_rgb_uint8: np.ndarray) -> bytes:
    pil = Image.fromarray(ensure_uint8_rgb(img_rgb_uint8), mode="RGB")
    bio = io.BytesIO()
    pil.save(bio, format="PNG")
    return bio.getvalue()


def pdf_safe(text) -> str:
    if text is None:
        return ""
    s = str(text)
    s = (s.replace("°", " deg")
           .replace("–", "-")
           .replace("—", "-")
           .replace("’", "'")
           .replace("“", '"')
           .replace("”", '"'))
    return s.encode("latin-1", errors="ignore").decode("latin-1")


def _to_float(val):
    if val is None:
        return None
    s = str(val).replace(",", ".")
    num = ""
    for ch in s:
        if ch.isdigit() or ch in ".-":
            num += ch
        elif num:
            break
    try:
        return float(num)
    except Exception:
        return None


def _badge(status: str):
    if status == "OK":
        return "🟢 OK"
    if status == "SURV":
        return "🟠 À surveiller"
    return "🔴 À corriger"


def _status_from_deg(deg: float):
    if deg is None:
        return "SURV"
    if deg < 2:
        return "OK"
    if deg < 5:
        return "SURV"
    return "ALERTE"


def calculate_angle(p1, p2, p3) -> float:
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)
    dot = float(np.dot(v1, v2))
    mag = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(np.clip(dot / mag, -1, 1)))


def angle_segment_vs_vertical(p1, p2) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))


def signed_angle_segment_vs_vertical(p1, p2) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return math.degrees(math.atan2(dx, abs(dy) + 1e-9))


def crop_to_landmarks(img_rgb_uint8: np.ndarray, res_pose, pad_ratio: float = 0.18) -> np.ndarray:
    if res_pose is None or not res_pose.pose_landmarks:
        return img_rgb_uint8
    h, w = img_rgb_uint8.shape[:2]
    xs, ys = [], []
    for lm in res_pose.pose_landmarks.landmark:
        if getattr(lm, "visibility", 1.0) < 0.2:
            continue
        xs.append(lm.x * w)
        ys.append(lm.y * h)
    if not xs or not ys:
        return img_rgb_uint8
    x1, x2 = max(0, int(min(xs))), min(w - 1, int(max(xs)))
    y1, y2 = max(0, int(min(ys))), min(h - 1, int(max(ys)))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(0, x1 - pad_x)
    x2 = min(w - 1, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(h - 1, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return img_rgb_uint8
    return img_rgb_uint8[y1:y2, x1:x2].copy()


def choose_visible_side(landmarks):
    L = mp_pose.PoseLandmark
    left_ids = [L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_HEEL, L.LEFT_EAR]
    right_ids = [L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_HEEL, L.RIGHT_EAR]

    left_score = sum(float(getattr(landmarks[i.value], "visibility", 0.0)) for i in left_ids)
    right_score = sum(float(getattr(landmarks[i.value], "visibility", 0.0)) for i in right_ids)
    return "left" if left_score >= right_score else "right"


def extract_points(img_rgb_uint8: np.ndarray):
    res = pose.process(img_rgb_uint8)
    if not res.pose_landmarks:
        return None, None
    lm = res.pose_landmarks.landmark
    side = choose_visible_side(lm)
    h, w = img_rgb_uint8.shape[:2]
    L = mp_pose.PoseLandmark

    def pt(enum_):
        p = lm[enum_.value]
        return np.array([p.x * w, p.y * h], dtype=np.float32)

    side_map = {
        "left": {
            "Epaule": pt(L.LEFT_SHOULDER),
            "Hanche": pt(L.LEFT_HIP),
            "Genou": pt(L.LEFT_KNEE),
            "Cheville": pt(L.LEFT_ANKLE),
            "Talon": pt(L.LEFT_HEEL),
            "Oreille": pt(L.LEFT_EAR),
        },
        "right": {
            "Epaule": pt(L.RIGHT_SHOULDER),
            "Hanche": pt(L.RIGHT_HIP),
            "Genou": pt(L.RIGHT_KNEE),
            "Cheville": pt(L.RIGHT_ANKLE),
            "Talon": pt(L.RIGHT_HEEL),
            "Oreille": pt(L.RIGHT_EAR),
        },
    }
    points = side_map[side]
    points["Nez"] = pt(L.NOSE)
    return side, points


def draw_preview(img_disp_rgb_uint8: np.ndarray, origin_points: dict, override_points: dict, scale: float) -> np.ndarray:
    out_bgr = cv2.cvtColor(img_disp_rgb_uint8.copy(), cv2.COLOR_RGB2BGR)
    for name, p in origin_points.items():
        x = int(p[0] * scale)
        y = int(p[1] * scale)
        cv2.circle(out_bgr, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(out_bgr, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    for name, p in override_points.items():
        x = int(p[0] * scale)
        y = int(p[1] * scale)
        cv2.circle(out_bgr, (x, y), 10, (255, 0, 255), 3)
        cv2.line(out_bgr, (x - 12, y), (x + 12, y), (255, 0, 255), 2)
        cv2.line(out_bgr, (x, y - 12), (x, y + 12), (255, 0, 255), 2)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def generate_pdf(data: dict, img_rgb_uint8: np.ndarray) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(31, 73, 125)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.set_y(10)
    pdf.cell(0, 10, "COMPTE-RENDU POSTURAL LATERAL", align="C", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_y(42)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(120, 8, pdf_safe(f"Patient : {data.get('Nom', '')}"), ln=0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(70, 8, datetime.now().strftime("%d/%m/%Y %H:%M"), ln=1, align="R")
    pdf.ln(2)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    tmp_img = os.path.join(tempfile.gettempdir(), "posture_lateral_tmp.png")
    Image.fromarray(img_rgb_uint8).save(tmp_img)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, "Photographie annotée", ln=True)
    y = pdf.get_y()
    page_w = pdf.w - 2 * pdf.l_margin
    avail_h = pdf.h - pdf.b_margin - y
    ih, iw = img_rgb_uint8.shape[:2]
    aspect = iw / ih
    target_w = page_w * 0.62
    target_h = target_w / aspect
    if target_h > avail_h:
        target_h = avail_h
        target_w = target_h * aspect
    x = (pdf.w - target_w) / 2
    pdf.image(tmp_img, x=x, y=y, w=target_w, h=target_h)
    pdf.set_y(y + target_h + 4)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, "Synthèse", ln=True)
    pdf.set_font("Arial", '', 11)
    for key in [
        "Côté détecté",
        "Inclinaison Jambe / verticale",
        "Inclinaison Cuisse / verticale",
        "Inclinaison Tronc / verticale",
        "Inclinaison Tête-Cou / verticale",
    ]:
        if key in data:
            pdf.cell(0, 6, pdf_safe(f"- {key} : {data[key]}"), ln=True)

    pdf.ln(3)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(120, 9, "Indicateur", 1, 0, 'L', True)
    pdf.cell(70, 9, "Valeur", 1, 1, 'C', True)
    pdf.set_font("Arial", '', 11)
    for k, v in data.items():
        if k != "Nom":
            pdf.cell(120, 8, pdf_safe(k), 1, 0, 'L')
            pdf.cell(70, 8, pdf_safe(v), 1, 1, 'C')

    pdf.set_y(-18)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, "Document indicatif - Ne remplace pas un avis médical.", align="C")

    try:
        os.remove(tmp_img)
    except Exception:
        pass

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1")


if "override_points" not in st.session_state:
    st.session_state["override_points"] = {}

with st.sidebar:
    st.header("👤 Dossier patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    taille_cm = st.number_input("Taille (cm)", min_value=100, max_value=230, value=170)
    source = st.radio("Source de l'image", ["📷 Caméra", "📁 Téléverser une photo"])
    st.divider()
    st.subheader("🖱️ Correction avant analyse")
    enable_click_edit = st.checkbox("Activer correction par clic", value=True)
    point_to_edit = st.selectbox(
        "Point à corriger",
        ["Epaule", "Hanche", "Genou", "Cheville", "Talon", "Oreille", "Nez"],
        disabled=not enable_click_edit,
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Reset point", disabled=not enable_click_edit):
            st.session_state["override_points"].pop(point_to_edit, None)
    with c2:
        if st.button("🧹 Reset tout", disabled=not enable_click_edit):
            st.session_state["override_points"] = {}
    st.divider()
    disp_w_user = st.slider("Largeur d'affichage (px)", 320, 900, 520, 10)
    auto_crop = st.checkbox("Cadrage automatique", value=True)

col_input, col_result = st.columns([1, 1])

with col_input:
    if source == "📷 Caméra":
        image_data = st.camera_input("Capturez la posture latérale")
    else:
        image_data = st.file_uploader("Format JPG/PNG", type=["jpg", "jpeg", "png"])

if not image_data:
    st.stop()

if isinstance(image_data, Image.Image):
    img_np = np.array(image_data.convert("RGB"))
else:
    img_np = np.array(Image.open(image_data).convert("RGB"))

img_np = ensure_uint8_rgb(rotate_if_landscape(img_np))
res_for_crop = pose.process(img_np)
if auto_crop:
    img_np = ensure_uint8_rgb(crop_to_landmarks(img_np, res_for_crop, pad_ratio=0.18))

h, w = img_np.shape[:2]
side_detected, origin_points = extract_points(img_np)
if origin_points is None:
    st.error("Aucune pose détectée. Utilisez une photo nette, de profil, en pied.")
    st.stop()

with col_input:
    st.subheader("📌 Cliquez pour corriger le point sélectionné")
    st.caption(f"Côté détecté automatiquement : **{'Gauche' if side_detected == 'left' else 'Droite'}**")
    disp_w = min(int(disp_w_user), w)
    scale = disp_w / w
    disp_h = int(h * scale)
    img_disp = cv2.resize(img_np, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    preview = draw_preview(img_disp, origin_points, st.session_state["override_points"], scale)
    coords = streamlit_image_coordinates(Image.open(io.BytesIO(to_png_bytes(preview))), key="img_click")
    if enable_click_edit and coords is not None:
        x_orig = float(coords["x"]) / scale
        y_orig = float(coords["y"]) / scale
        st.session_state["override_points"][point_to_edit] = (x_orig, y_orig)
        st.success(f"✅ {point_to_edit} placé à ({x_orig:.0f}, {y_orig:.0f}) px")
    if st.session_state["override_points"]:
        st.write("**Point(s) corrigé(s) :**")
        for k, (x, y) in st.session_state["override_points"].items():
            st.write(f"- {k} → ({x:.0f}, {y:.0f})")

with col_result:
    st.subheader("⚙️ Analyse")
    run = st.button("▶ Lancer l'analyse")

if not run:
    st.stop()

points = {k: v.copy() for k, v in origin_points.items()}
for k, v in st.session_state["override_points"].items():
    if k in points:
        points[k] = np.array([v[0], v[1]], dtype=np.float32)

Epaule = points["Epaule"]
Hanche = points["Hanche"]
Genou = points["Genou"]
Cheville = points["Cheville"]
Talon = points["Talon"]
Oreille = points["Oreille"]
Nez = points["Nez"]

incl_jambe = angle_segment_vs_vertical(Genou, Cheville)
incl_cuisse = angle_segment_vs_vertical(Hanche, Genou)
incl_tronc = angle_segment_vs_vertical(Hanche, Epaule)
incl_tete_cou = angle_segment_vs_vertical(Epaule, Oreille)
incl_tete_nez = angle_segment_vs_vertical(Oreille, Nez)

signed_tronc = signed_angle_segment_vs_vertical(Hanche, Epaule)
signed_tete = signed_angle_segment_vs_vertical(Epaule, Oreille)
angle_genou = calculate_angle(Hanche, Genou, Cheville)
angle_cheville = calculate_angle(Genou, Cheville, Talon)

sens_tronc = "Vers l'avant" if signed_tronc > 0 else "Vers l'arrière"
sens_tete = "Vers l'avant" if signed_tete > 0 else "Vers l'arrière"

results = {
    "Nom": nom,
    "Plan": "Latéral",
    "Côté détecté": "Gauche" if side_detected == "left" else "Droite",
    "Inclinaison Jambe / verticale": f"{incl_jambe:.1f}°",
    "Inclinaison Cuisse / verticale": f"{incl_cuisse:.1f}°",
    "Inclinaison Tronc / verticale": f"{incl_tronc:.1f}°",
    "Sens inclinaison tronc": sens_tronc,
    "Inclinaison Tête-Cou / verticale": f"{incl_tete_cou:.1f}°",
    "Sens inclinaison tête": sens_tete,
    "Inclinaison Tête (oreille-nez)": f"{incl_tete_nez:.1f}°",
    "Angle Genou": f"{angle_genou:.1f}°",
    "Angle Cheville": f"{angle_cheville:.1f}°",
}

ann_bgr = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)
for _, p in points.items():
    cv2.circle(ann_bgr, tuple(np.round(p).astype(int)), 7, (0, 255, 0), -1)
for name, p in st.session_state["override_points"].items():
    arr = np.array(p)
    cv2.circle(ann_bgr, tuple(np.round(arr).astype(int)), 14, (255, 0, 255), 3)
    cv2.putText(ann_bgr, name, (int(arr[0]) + 8, int(arr[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
cv2.line(ann_bgr, tuple(np.round(Hanche).astype(int)), tuple(np.round(Epaule).astype(int)), (255, 0, 0), 3)
cv2.line(ann_bgr, tuple(np.round(Hanche).astype(int)), tuple(np.round(Genou).astype(int)), (0, 200, 0), 3)
cv2.line(ann_bgr, tuple(np.round(Genou).astype(int)), tuple(np.round(Cheville).astype(int)), (0, 255, 255), 3)
cv2.line(ann_bgr, tuple(np.round(Epaule).astype(int)), tuple(np.round(Oreille).astype(int)), (255, 0, 255), 3)
cv2.line(ann_bgr, tuple(np.round(Oreille).astype(int)), tuple(np.round(Nez).astype(int)), (100, 255, 100), 2)
annotated = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
annotated = ensure_uint8_rgb(annotated)

with col_result:
    st.subheader("🧾 Compte-rendu d'analyse posturale")
    st.markdown("### 🧑‍⚕️ Identité")
    st.write(f"**Patient :** {nom}")
    st.write(f"**Taille déclarée :** {taille_cm} cm")
    st.write(f"**Date/heure :** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.write(f"**Côté détecté :** {results['Côté détecté']}")
    st.markdown("---")

    leg_deg = _to_float(results.get("Inclinaison Jambe / verticale"))
    thigh_deg = _to_float(results.get("Inclinaison Cuisse / verticale"))
    trunk_deg = _to_float(results.get("Inclinaison Tronc / verticale"))
    head_deg = _to_float(results.get("Inclinaison Tête-Cou / verticale"))

    st.markdown("### 📌 Synthèse latérale")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Jambe / verticale**")
        st.write(results["Inclinaison Jambe / verticale"])
        st.write(_badge(_status_from_deg(leg_deg)))
    with c2:
        st.markdown("**Cuisse / verticale**")
        st.write(results["Inclinaison Cuisse / verticale"])
        st.write(_badge(_status_from_deg(thigh_deg)))
    with c3:
        st.markdown("**Tronc / verticale**")
        st.write(results["Inclinaison Tronc / verticale"])
        st.write(_badge(_status_from_deg(trunk_deg)))
    with c4:
        st.markdown("**Tête / verticale**")
        st.write(results["Inclinaison Tête-Cou / verticale"])
        st.write(_badge(_status_from_deg(head_deg)))

    st.markdown("### 🧩 Détails")
    for key in [
        "Inclinaison Jambe / verticale",
        "Inclinaison Cuisse / verticale",
        "Inclinaison Tronc / verticale",
        "Sens inclinaison tronc",
        "Inclinaison Tête-Cou / verticale",
        "Sens inclinaison tête",
        "Inclinaison Tête (oreille-nez)",
        "Angle Genou",
        "Angle Cheville",
    ]:
        st.write(f"- {key} : {results[key]}")

    st.markdown("### ✅ Observations automatiques")
    obs = []
    if trunk_deg is not None:
        obs.append("Tronc : alignement satisfaisant." if trunk_deg < 2 else "Tronc : légère inclinaison sagittale." if trunk_deg < 5 else "Tronc : inclinaison marquée.")
    if head_deg is not None:
        obs.append("Tête/cou : alignement satisfaisant." if head_deg < 2 else "Tête/cou : légère projection ou inclinaison." if head_deg < 5 else "Tête/cou : désalignement marqué.")
    if leg_deg is not None:
        obs.append("Jambe : orientation proche de la verticale." if leg_deg < 2 else "Jambe : légère inclinaison sagittale." if leg_deg < 5 else "Jambe : inclinaison marquée.")
    for o in obs:
        st.write(f"- {o}")

    st.markdown("### 📝 Tableau des mesures")
    st.table(results)

    st.markdown("### 🖼️ Image annotée")
    st.image(annotated, caption="Points verts = utilisés | Violet = corrigé", use_column_width=True)

    st.markdown("---")
    st.subheader("📄 PDF")
    pdf_bytes = generate_pdf(results, annotated)
    pdf_name = f"Bilan_Lateral_{pdf_safe(results.get('Nom', 'Anonyme')).replace(' ', '_')}.pdf"
    st.download_button(
        label="📥 Télécharger le Bilan PDF",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
    )
