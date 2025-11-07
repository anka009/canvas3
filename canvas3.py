# canvas2_beautified.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import json
from pathlib import Path

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_points(points, min_dist=6):
    out = []
    for p in points:
        if not any(is_near(p, q, r=min_dist) for q in out):
            out.append(p)
    return out

def get_centers(mask, min_area=50):
    m = mask.copy()
    res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[1] if len(res) > 1 else []
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00", 0):
                cx = int(round(M["m10"] / M["m00"]))
                cy = int(round(M["m01"] / M["m00"]))
                centers.append((cx, cy))
    return centers

def compute_hsv_range(points, hsv_img, radius=5):
    if not points: return None
    vals = []
    for (x, y) in points:
        x_min = max(0, x - radius)
        x_max = min(hsv_img.shape[1], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(hsv_img.shape[0], y + radius + 1)
        region = hsv_img[y_min:y_max, x_min:x_max]
        if region.size > 0:
            vals.append(region.reshape(-1, 3))
    if not vals: return None
    vals = np.vstack(vals)
    h = vals[:,0].astype(int)
    s = vals[:,1].astype(int)
    v = vals[:,2].astype(int)
    h_med = float(np.median(h))
    s_med = float(np.median(s))
    v_med = float(np.median(v))
    n_points = len(points)
    tol_h = int(min(25, 10 + n_points * 3))
    tol_s = int(min(80, 30 + n_points * 10))
    tol_v = int(min(80, 30 + n_points * 10))
    if np.mean(h) > 150 or np.mean(h) < 20:
        h_med = float(np.median(np.where(h < 90, h + 180, h)) % 180)
        tol_h = min(40, tol_h + 5)
    h_min = int(round((h_med - tol_h) % 180))
    h_max = int(round((h_med + tol_h) % 180))
    s_min = max(0, int(round(s_med - tol_s)))
    s_max = min(255, int(round(s_med + tol_s)))
    v_min = max(0, int(round(v_med - tol_v)))
    v_max = min(255, int(round(v_med + tol_v)))
    return (h_min, h_max, s_min, s_max, v_min, v_max)

def apply_hue_wrap(hsv_img, hmin, hmax, smin, smax, vmin, vmax):
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([hmax,smax,vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0,smin,vmin]), np.array([hmax,smax,vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([180,smax,vmax]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)
    return mask

def ensure_odd(k):
    return k if k % 2 == 1 else k + 1

def remove_near(points, forbidden_points, r):
    if not forbidden_points: return points
    return [p for p in points if not any(is_near(p, q, r) for q in forbidden_points)]

def save_last_calibration(path="kalibrierung.json"):
    data = {
        "aec_hsv": st.session_state.get("aec_hsv").tolist() if st.session_state.get("aec_hsv") is not None else None,
        "hema_hsv": st.session_state.get("hema_hsv").tolist() if st.session_state.get("hema_hsv") is not None else None,
        "bg_hsv": st.session_state.get("bg_hsv").tolist() if st.session_state.get("bg_hsv") is not None else None
    }
    try:
        with open(path,"w") as f:
            json.dump(data,f)
        st.success("💾 Kalibrierung gespeichert.")
    except Exception as e:
        st.error(f"Fehler beim Speichern: {e}")

def load_last_calibration(path="kalibrierung.json"):
    try:
        with open(path,"r") as f:
            data = json.load(f)
        st.session_state.aec_hsv = np.array(data.get("aec_hsv")) if data.get("aec_hsv") else None
        st.session_state.hema_hsv = np.array(data.get("hema_hsv")) if data.get("hema_hsv") else None
        st.session_state.bg_hsv = np.array(data.get("bg_hsv")) if data.get("bg_hsv") else None
        st.success("✅ Letzte Kalibrierung geladen.")
    except FileNotFoundError:
        st.warning("⚠️ Keine gespeicherte Kalibrierung gefunden.")
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler (Auto-Kalib)", layout="wide")
st.title("🧬 Zellkern-Zähler – Auto-Kalibrierung (AEC / Hämatoxylin)")

# -------------------- Session State --------------------
keys_defaults = [
    "aec_cal_points","hema_cal_points","bg_cal_points",
    "aec_auto","hema_auto",
    "manual_aec","manual_hema",
    "aec_hsv","hema_hsv","bg_hsv",
    "last_file","disp_width","last_auto_run"
]
for k in keys_defaults:
    if k not in st.session_state:
        st.session_state[k] = [] if "points" in k or "auto" in k or "manual" in k else None if "hsv" in k else 1400 if k=="disp_width" else 0

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen."); st.stop()

if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_cal_points","hema_cal_points","bg_cal_points","aec_auto","hema_auto","manual_aec","manual_hema"]:
        st.session_state[k] = []
    for k in ["aec_hsv","hema_hsv","bg_hsv"]:
        st.session_state[k] = None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400
    st.session_state.last_auto_run = 0

# -------------------- Bild vorbereiten --------------------
colW1,colW2 = st.columns([2,1])
with colW1:
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400,2000,st.session_state.disp_width, step=100)
    st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig,W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH/W_orig
image_disp = cv2.resize(image_orig,(DISPLAY_WIDTH,int(H_orig*scale)),interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# -------------------- Sidebar --------------------
st.sidebar.markdown("### ⚙️ Filter & Kalibrierung")
blur_kernel = ensure_odd(st.sidebar.slider("🔧 Blur",1,21,5,1))
min_area = st.sidebar.number_input("📏 Mindestfläche (px)",10,2000,100)
alpha = st.sidebar.slider("🌗 Alpha (Kontrast)",0.1,3.0,1.0,0.1)
circle_radius = st.sidebar.slider("⚪ Kreisradius (Display-Px)",1,20,5)
calib_radius = st.sidebar.slider("🎯 Kalibrierungsradius (Pixel)",1,15,5)
min_points_calib = st.sidebar.slider("🧮 Min Punkte für Auto-Kalibrierung",1,10,3)
st.sidebar.info("Kalibrierung läuft automatisch sobald min Punkte erreicht.")

# Mode selection
st.sidebar.markdown("### 🎨 Modus")
mode = st.sidebar.radio("Modus", [
    "Keine",
    "AEC Kalibrier-Punkt setzen",
    "Hämatoxylin Kalibrier-Punkt setzen",
    "Hintergrund Kalibrier-Punkt setzen",
    "AEC manuell hinzufügen",
    "Hämatoxylin manuell hinzufügen",
    "Punkt löschen"
], index=0)

aec_mode = mode=="AEC Kalibrier-Punkt setzen"
hema_mode = mode=="Hämatoxylin Kalibrier-Punkt setzen"
bg_mode = mode=="Hintergrund Kalibrier-Punkt setzen"
manual_aec_mode = mode=="AEC manuell hinzufügen"
manual_hema_mode = mode=="Hämatoxylin manuell hinzufügen"
delete_mode = mode=="Punkt löschen"

# -------------------- Schnellaktionen --------------------
st.sidebar.markdown("### ⚡ Schnellaktionen")
if st.sidebar.button("🧹 Alles löschen"):
    for k in ["aec_cal_points","hema_cal_points","bg_cal_points","aec_auto","hema_auto","manual_aec","manual_hema","aec_hsv","hema_hsv","bg_hsv"]:
        st.session_state[k] = [] if "points" in k or "auto" in k or "manual" in k else None
    st.success("Alles gelöscht ✅")

# ... Klicklogik, Auto-Kalibrierung, Anzeige usw. bleibt identisch ...
