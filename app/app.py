import sys
import os
import time
from PIL import Image
import numpy as np
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_image

MODEL_NAME = "MobileNetV2 (Feature Extraction)"
INPUT_SIZE = "224 × 224 × 3"
DATASET_NAME = "Chest X-Ray Pneumonia (5,863 images)"
ALLOWED_TYPES = ["jpeg", "jpg", "png", "bmp"]
BORDER_MARGIN = 0.10


def input_warnings(img):
    arr = np.asarray(img.resize((128, 128)), dtype=np.float32)
    colorfulness = float(
        np.mean(np.abs(arr[..., 0] - arr[..., 1]) + np.abs(arr[..., 1] - arr[..., 2]))
    )
    warns = []
    if colorfulness > 12:
        warns.append("COLOR_IMAGE: NOT A TYPICAL X-RAY")
    if min(img.size) < 150:
        warns.append("LOW_RESOLUTION: RESULT UNRELIABLE")
    return warns


def inject_html(block: str) -> None:
    cleaned = "\n".join(line.strip() for line in block.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


st.set_page_config(
    page_title="CHESTSCAN AI | NEUROTRACE",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_html("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=Space+Mono:wght@400;700&display=swap');

/* Les noms de variables sont conservés car le HTML inline les utilise ; seules les valeurs changent. */
:root {
    --bg-color: #0a1020;      /* fond de page */
    --text-black: #e8eefc;    /* texte principal (clair sur fond sombre) */
    --accent-red: #ff5c7a;    /* alertes : pneumonie */
    --border-dark: #26334f;   /* bordures */
    --white: #07101f;         /* texte posé sur un fond plein (boutons, pastilles) */
    --brand: #2ee6d6;         /* couleur de marque : turquoise */
    --brand-2: #5b8cff;       /* second ton du dégradé */
    --warn: #ffc857;          /* zone incertaine, avertissements */
    --panel: #111a30;         /* cartes */
    --muted: #8da2c8;         /* texte secondaire */
}

html, body { background-color: var(--bg-color) !important; }
html, body, [class*="css"], .stApp {
    font-family: 'Sora', sans-serif;
    color: var(--text-black);
}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p { color: var(--text-black); }

.stApp {
    background:
        radial-gradient(900px 520px at 85% -10%, rgba(46,230,214,0.14), transparent 60%),
        radial-gradient(760px 520px at -10% 30%, rgba(91,140,255,0.14), transparent 60%),
        var(--bg-color) !important;
}
.stApp::before {
    content: "";
    position: fixed; inset: 0;
    pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(140,170,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(140,170,255,0.05) 1px, transparent 1px);
    background-size: 56px 56px;
    -webkit-mask-image: radial-gradient(ellipse at 50% 0%, #000 30%, transparent 75%);
    mask-image: radial-gradient(ellipse at 50% 0%, #000 30%, transparent 75%);
}

[data-testid="stSidebar"], header[data-testid="stHeader"], footer { display: none !important; }
.block-container { padding: 2rem 4rem; max-width: 1600px; position: relative; z-index: 1; }
.block-container::before, .block-container::after { content: none; }

.top-nav { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 1px solid var(--border-dark); padding-bottom: 14px; }
.logo-pill { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 18px; letter-spacing: 3px; display: flex; align-items: center; gap: 10px; }
.logo-pill i { color: var(--brand); filter: drop-shadow(0 0 8px rgba(46,230,214,0.6)); }
.status-pill { font-family: 'Space Mono', monospace; font-weight: 700; font-size: 12px; display: flex; align-items: center; gap: 10px; border: 1px solid rgba(46,230,214,0.45); color: var(--brand); padding: 6px 16px; border-radius: 999px; background: rgba(46,230,214,0.07); }
.blink-dot { width: 9px; height: 9px; background-color: var(--brand); border-radius: 50%; box-shadow: 0 0 10px var(--brand); animation: blink 1.5s infinite; }
@keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.15; } 100% { opacity: 1; } }

.hero-section { position: relative; margin-bottom: 40px; }
.giant-title, h1.giant-title {
    font-family: 'Sora', sans-serif !important;
    font-size: clamp(48px, 9vw, 122px) !important; font-weight: 800 !important; line-height: 0.95 !important; letter-spacing: -0.04em !important;
    margin: 0 !important; padding: 0 !important; text-transform: uppercase;
    background: linear-gradient(90deg, #e8eefc 0%, var(--brand) 60%, var(--brand-2) 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
}
.barcode { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 2px; position: absolute; right: 0; bottom: 10px; text-align: right; color: var(--muted); }
.barcode-lines { display: inline-block; width: 120px; height: 26px; opacity: 0.55; background: repeating-linear-gradient(90deg, var(--brand) 0, var(--brand) 2px, transparent 2px, transparent 4px, var(--brand) 4px, var(--brand) 5px, transparent 5px, transparent 8px); }

.tech-box { border: 1px solid var(--border-dark); background: var(--panel); border-radius: 14px; padding: 22px; position: relative; margin-bottom: 20px; box-shadow: 0 12px 32px rgba(0,0,0,0.35); }
.box-header { font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 700; text-transform: uppercase; background: var(--text-black); color: var(--white); padding: 4px 12px; border-radius: 6px; display: inline-block; position: absolute; top: -14px; left: 15px; letter-spacing: 1px; }

.st-key-input_box {
    position: relative;
    background: var(--panel);
    border: 1px solid var(--border-dark);
    border-radius: 14px;
    padding: 28px 20px 20px !important;
    margin-top: 18px;
    margin-bottom: 20px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.35);
}
.st-key-input_box::before {
    content: 'INPUT_MODULE';
    position: absolute; top: -14px; left: 15px;
    font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 700; letter-spacing: 1px;
    background: var(--brand); color: var(--white); padding: 4px 12px; border-radius: 6px;
}

.st-key-img_box {
    position: relative;
    border: 1px solid rgba(46,230,214,0.4);
    border-radius: 12px;
    padding: 6px !important;
    background: #050a15;
    overflow: hidden;
    margin-top: 5px;
}
.st-key-img_box::before {
    content: '';
    position: absolute; left: 0; width: 100%; height: 3px; top: 0;
    background: linear-gradient(90deg, transparent, var(--brand), transparent);
    box-shadow: 0 0 18px var(--brand), 0 0 36px rgba(46,230,214,0.5);
    z-index: 5; pointer-events: none;
    animation: scan 2.5s infinite linear;
}
@keyframes scan { 0% { top: 0; opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { top: 98%; opacity: 0; } }
[data-testid="stImage"] img { filter: grayscale(100%) contrast(1.1); display: block; width: 100%; height: auto; border-radius: 8px; }

.stElementContainer:has([data-testid="stImage"]) { position: relative; }
[data-testid="stElementToolbar"] {
    top: auto !important;
    bottom: 12px !important;
    right: 12px !important;
    z-index: 20 !important;
}
[data-testid="stElementToolbar"] button,
[data-testid="StyledFullScreenButton"] {
    background-color: rgba(7, 16, 31, 0.8) !important;
    border-radius: 8px !important;
    color: var(--text-black) !important;
}
[data-testid="stElementToolbar"] button *,
[data-testid="StyledFullScreenButton"] * { color: var(--text-black) !important; fill: var(--text-black) !important; }
[data-testid="StyledFullScreenButton"] { top: auto !important; bottom: 12px !important; right: 12px !important; z-index: 20 !important; }
[data-testid="stElementToolbar"] button:hover,
[data-testid="StyledFullScreenButton"]:hover { background-color: var(--brand) !important; }
[data-testid="stElementToolbar"] button:hover *,
[data-testid="StyledFullScreenButton"]:hover * { color: var(--white) !important; fill: var(--white) !important; }

.result-huge { font-family: 'Sora', sans-serif; font-size: 84px; font-weight: 800; line-height: 1; margin: 10px 0; letter-spacing: -3px; }
.result-label { font-family: 'Space Mono', monospace; font-size: 16px; font-weight: 700; letter-spacing: 3px; }
.alert-red { color: var(--accent-red); text-shadow: 0 0 24px rgba(255,92,122,0.45); }
.ok-black { color: var(--brand); text-shadow: 0 0 24px rgba(46,230,214,0.4); }

[data-testid="stFileUploader"] { margin-top: 5px; }
[data-testid="stFileUploader"] section { background: rgba(46,230,214,0.04) !important; border: 1.5px dashed rgba(46,230,214,0.45) !important; border-radius: 12px !important; padding: 16px !important; margin-bottom: 0px !important; }
[data-testid="stFileUploader"] section * { color: var(--text-black) !important; opacity: 1 !important; visibility: visible !important; }
[data-testid="stFileUploader"] small { color: var(--muted) !important; font-weight: 600 !important; }
[data-testid="stFileUploader"] span { font-weight: 600 !important; }
[data-testid="stFileUploader"] section button,
[data-testid="stFileUploader"] section button *,
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button * {
    color: var(--white) !important;
    -webkit-text-fill-color: var(--white) !important;
}
[data-testid="stFileUploader"] section button,
[data-testid="stFileUploaderDropzone"] button {
    background: var(--brand) !important;
    font-family: 'Space Mono', monospace; font-size: 11px !important; font-weight: 700; border-radius: 10px !important;
    border: 1px solid var(--brand) !important;
}
[data-testid="stFileUploader"] section button:hover,
[data-testid="stFileUploaderDropzone"] button:hover { background: #7cf5ea !important; border-color: #7cf5ea !important; }

[data-testid="stSlider"] { padding-top: 5px; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--brand); font-weight: bold; margin-bottom: 5px; }
[data-testid="stSlider"] [role="slider"] { background: var(--brand); box-shadow: 0 0 0 4px rgba(46,230,214,0.18); }

.stButton, .stElementContainer:has(.stButton), [data-testid="stButton"] { width: 100% !important; }
.stButton:first-of-type > button { width: 100% !important; display: block; background: linear-gradient(135deg, var(--brand), var(--brand-2)); color: var(--white); border: none; border-radius: 14px; padding: 18px 20px; font-weight: 800; font-size: 17px; letter-spacing: 2px; text-transform: uppercase; transition: all 0.2s ease; margin-bottom: 20px; box-shadow: 0 10px 28px rgba(46,230,214,0.25); }
.stButton:first-of-type > button p, .stButton:first-of-type > button [data-testid="stMarkdownContainer"] p { color: var(--white) !important; }
.stButton:first-of-type > button:hover { filter: brightness(1.08); transform: translateY(-2px); box-shadow: 0 14px 34px rgba(46,230,214,0.4); color: var(--white); }
.stButton:first-of-type > button:disabled { opacity: 0.35; filter: grayscale(0.6); box-shadow: none; cursor: not-allowed; transform: none; }

.terminal-log { font-family: 'Space Mono', monospace; font-size: 11px; color: var(--muted); border-left: 2px solid var(--brand); padding-left: 15px; line-height: 1.8; margin-top: 15px; }
.terminal-log b { color: var(--text-black); }

.borderline { border: 2px dashed var(--warn) !important; }
.warn-line { font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 700; color: var(--warn); margin-top: 8px; }

.class-bars { margin-top: 22px; }
.cb-row { margin-bottom: 16px; }
.cb-head { display:flex; justify-content:space-between; font-family:'Space Mono', monospace; font-size:12px; font-weight:700; margin-bottom:6px; }
.cb-track { position:relative; height:14px; border-radius:999px; border:1px solid var(--border-dark); background: rgba(255,255,255,0.05); }
.cb-fill { height:100%; border-radius:999px; }
.cb-fill.dark { background: linear-gradient(90deg, #14b8a6, var(--brand)); }
.cb-fill.red { background: linear-gradient(90deg, #ff8a5c, var(--accent-red)); }
.cb-thr { position:absolute; top:-5px; bottom:-5px; width:0; border-left:2px dashed var(--warn); }
.cb-thr em { position:absolute; top:-16px; left:4px; font-family:'Space Mono', monospace; font-size:9px; font-style:normal; font-weight:700; color:var(--warn); white-space:nowrap; }
.cb-pred { background: var(--brand); color: var(--white); padding: 2px 8px; margin-left: 8px; font-size: 10px; border-radius: 6px; }

[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFile"] small {
    color: var(--text-black) !important;
    -webkit-text-fill-color: var(--text-black) !important;
}

[data-testid="stSpinner"], [data-testid="stSpinner"] * { color: var(--text-black) !important; font-family: 'Space Mono', monospace; }
[data-testid="stAlert"] { background: rgba(255,92,122,0.12) !important; border: 1px solid rgba(255,92,122,0.4); border-radius: 12px; }
[data-testid="stAlert"] * { color: var(--text-black) !important; }
</style>
""")

inject_html("""
<div class="top-nav">
    <div class="logo-pill"><i class="fa-solid fa-lungs"></i> CHESTSCAN</div>
    <div class="status-pill">SYS. ACTIVE <div class="blink-dot"></div></div>
</div>
<div class="hero-section">
    <h1 class="giant-title">CHESTSCAN</h1>
    <div class="barcode">
        <div class="barcode-lines"></div><br>
        UID: 994-CX-8820
    </div>
    <div class="terminal-log" style="max-width: 500px; margin-top: 20px; border-left-color: var(--border-dark);">
        > INITIATING PROTOCOL_01<br>
        > AI-DRIVEN RADIOLOGY ANALYSIS ENGINE<br>
        > DETECTING PULMONARY OPACITIES WITH HIGH PRECISION.
    </div>
</div>
""")

col_in, col_out = st.columns([1.1, 1], gap="large")

with col_in:
    image = None

    with st.container(key="input_box"):
        uploaded_file = st.file_uploader(
            "Importer une radiographie",
            type=ALLOWED_TYPES,
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image.load()
                image = image.convert("RGB")
            except Exception as exc:
                image = None
                st.error(f"SYSTEM ERROR: {exc}")

        if image is not None:
            inject_html(f"""
<div style="font-family:'Space Mono'; font-size:10px; color:var(--text-black); padding-top:6px; opacity:0.8; margin-bottom: 5px;">
RES: {image.width}x{image.height}
</div>
""")
            with st.container(key="img_box"):
                st.image(image, use_container_width=True)

    inject_html(
        "<div style=\"font-family:'Space Mono'; font-size:11px; font-weight:bold; margin-bottom:5px;\">DETECTION_THRESHOLD</div>"
    )
    threshold = st.slider(
        "DETECTION_THRESHOLD",
        min_value=0.50,
        max_value=0.80,
        value=0.65,
        step=0.01,
        label_visibility="collapsed",
    )

with col_out:
    run_analysis = st.button("EXECUTE SCAN", disabled=(image is None))

    file_key = (
        (uploaded_file.name, uploaded_file.size) if uploaded_file is not None else None
    )

    if run_analysis and image is not None:
        with st.spinner("EXTRACTING FEATURES..."):
            try:
                t0 = time.perf_counter()
                res = predict_image(image, threshold=threshold)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                st.session_state["scan"] = {
                    "key": file_key,
                    "prob": float(res["probability"]),
                    "ms": res.get("inference_ms", elapsed_ms),
                }
            except Exception as e:
                st.session_state.pop("scan", None)
                st.error(f"CRITICAL FAILURE: {e}")

    scan = st.session_state.get("scan")
    ms_txt = "—"
    if scan is not None and image is not None and scan["key"] == file_key:
        prob = scan["prob"]
        pred = "PNEUMONIA" if prob >= threshold else "NORMAL"
        is_pneu = pred == "PNEUMONIA"
        borderline = abs(prob - threshold) <= BORDER_MARGIN
        if scan["ms"] is not None:
            ms_txt = f"{scan['ms']:.0f} ms"

        text_color = "alert-red" if is_pneu else "ok-black"
        ico = "fa-triangle-exclamation" if is_pneu else "fa-check-double"
        border_color = "var(--accent-red)" if is_pneu else "var(--text-black)"

        if borderline:
            label = f"UNCERTAIN ZONE // LEANS {pred}"
            ico = "fa-circle-question"
            box_cls = "tech-box borderline"
            action = "> ACTION: BORDERLINE CASE, REVIEW MANUALLY<br>"
        else:
            label = pred
            box_cls = "tech-box"
            action = ""

        warn_html = "".join(
            f'<div class="warn-line">! {w}</div>' for w in input_warnings(image)
        )

        p_pneu = prob
        p_norm = 1.0 - prob
        conf = p_pneu if is_pneu else p_norm

        tag = '<span class="cb-pred">PREDICTED</span>'
        inject_html(f"""
            <div class="{box_cls}" style="border: 2px solid {border_color}; padding: 30px;">
                <div class="box-header" style="background:{border_color};">OUTPUT_RESULT</div>
                <div class="result-label {text_color}"><i class="fa-solid {ico}"></i> {label}</div>
                <div class="result-huge {text_color}">{conf * 100:.1f}%</div>
                <div class="result-label {text_color}" style="font-size:12px; letter-spacing:2px;">{pred} CONFIDENCE</div>
                <div class="class-bars">
                    <div class="cb-row">
                        <div class="cb-head"><span>NORMAL{tag if not is_pneu else ''}</span><span>{p_norm * 100:.1f}%</span></div>
                        <div class="cb-track"><div class="cb-fill dark" style="width:{p_norm * 100:.1f}%"></div></div>
                    </div>
                    <div class="cb-row">
                        <div class="cb-head"><span>PNEUMONIA{tag if is_pneu else ''}</span><span>{p_pneu * 100:.1f}%</span></div>
                        <div class="cb-track">
                            <div class="cb-fill red" style="width:{p_pneu * 100:.1f}%"></div>
                            <div class="cb-thr" style="left:{threshold * 100:.0f}%"><em>THR {threshold:.2f}</em></div>
                        </div>
                    </div>
                </div>
                <div class="terminal-log" style="border-left-color: {border_color}; margin-top: 24px;">
                    > NORMAL_SCORE: {p_norm:.4f}<br>
                    > PNEUMONIA_SCORE: {p_pneu:.4f}<br>
                    > THRESHOLD_APPLIED: {threshold:.2f} (PNEUMONIA IF SCORE >= THR)<br>
                    > DISTANCE_TO_THRESHOLD: {abs(prob - threshold):.3f}<br>
                    > INFERENCE_TIME: {ms_txt}<br>
                    {action}
                    > TARGET_CLASS: {pred}
                </div>
                {warn_html}
            </div>
            """)

    inject_html(f"""
        <div class="tech-box" style="margin-top: 20px; opacity: 0.9;">
            <div class="box-header">SYS_TELEMETRY</div>
            <div class="terminal-log" style="border:none; padding:0; margin:0; line-height:2;">
                <b>ARCHITECTURE :</b> {MODEL_NAME}<br>
                <b>INPUT_TENSOR :</b> {INPUT_SIZE}<br>
                <b>LOSS_FUNCTION:</b> Binary Cross-Entropy<br>
                <b>BASE_WEIGHTS :</b> ImageNet<br>
                <b>TRAINING_SET :</b> {DATASET_NAME}<br>
                <b>LAST_LATENCY :</b> {ms_txt}<br>
                <br>
                [ CPU MEMORY ] : STABLE<br>
                [ GPU STATUS ] : STANDBY
            </div>
        </div>
        """)

inject_html("""
    <div style="margin-top: 60px; padding-top: 15px; border-top: 1px solid var(--border-dark); display: flex; justify-content: space-between; font-family: 'Space Mono', monospace; font-size: 10px;">
        <span>// CHESTSCAN AI PROTOTYPE v2.0</span>
        <span>STRICTLY NOT FOR CLINICAL DIAGNOSIS</span>
    </div>
    """)