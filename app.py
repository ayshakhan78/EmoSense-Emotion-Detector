import streamlit as st
import joblib
import numpy as np
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmoSense · Emotion Detector",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0a0f;
    color: #f0eee8;
}

/* Hide Streamlit default chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #e8d5b7 0%, #f5a623 50%, #e8d5b7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    letter-spacing: -1px;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #6b6b7a;
    margin-top: 6px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 300;
}

.accuracy-badge {
    display: inline-block;
    background: rgba(245, 166, 35, 0.12);
    border: 1px solid rgba(245, 166, 35, 0.3);
    color: #f5a623;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    margin-top: 10px;
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #2a2a3a, transparent);
    margin: 28px 0;
}

/* Text area */
.stTextArea textarea {
    background-color: #13131e !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 14px !important;
    color: #f0eee8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 16px !important;
    line-height: 1.7 !important;
    transition: border-color 0.3s ease !important;
}
.stTextArea textarea:focus {
    border-color: #f5a623 !important;
    box-shadow: 0 0 0 3px rgba(245, 166, 35, 0.1) !important;
}
.stTextArea textarea::placeholder {
    color: #3a3a4a !important;
}
.stTextArea label {
    color: #8888a0 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #f5a623, #e8831a) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em !important;
    padding: 14px 28px !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(245, 166, 35, 0.35) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Result card */
.result-card {
    border-radius: 20px;
    padding: 36px 32px;
    margin: 24px 0;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.5s ease forwards;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0);    }
}

.result-emoji {
    font-size: 4rem;
    display: block;
    margin-bottom: 10px;
    animation: bounceIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
}
@keyframes bounceIn {
    from { transform: scale(0.3); opacity: 0; }
    to   { transform: scale(1);   opacity: 1; }
}

.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1;
}

.result-confidence {
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 8px;
    opacity: 0.75;
}

/* Emotion bar section */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a4a5a;
    margin-bottom: 16px;
    margin-top: 28px;
}

.emotion-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}
.emotion-name {
    width: 80px;
    font-size: 0.85rem;
    color: #8888a0;
    text-align: right;
    flex-shrink: 0;
    font-weight: 400;
}
.bar-bg {
    flex: 1;
    height: 6px;
    background: #1a1a28;
    border-radius: 10px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 1s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}
.pct-label {
    width: 40px;
    font-size: 0.8rem;
    color: #4a4a5a;
    text-align: right;
    flex-shrink: 0;
}

/* Warning */
.stAlert {
    background-color: #1e1a0e !important;
    border: 1px solid rgba(245, 166, 35, 0.3) !important;
    border-radius: 12px !important;
    color: #f5a623 !important;
}

/* Footer */
.footer-text {
    text-align: center;
    font-size: 0.72rem;
    color: #2a2a3a;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 40px;
    padding-bottom: 20px;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #f5a623 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model & vectorizer ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    cv = joblib.load('bow_vectorizer.pkl')
    model = joblib.load('lr_model_bow.pkl')
    return cv, model

cv, model = load_model()

# ── Emotion config ────────────────────────────────────────────────────────────
emotion_config = {
    "joy":      {"emoji": "😄", "color": "#f5c842", "bg": "rgba(245,200,66,0.08)",  "bar": "#f5c842"},
    "sadness":  {"emoji": "😢", "color": "#4a9eff", "bg": "rgba(74,158,255,0.08)",  "bar": "#4a9eff"},
    "anger":    {"emoji": "😠", "color": "#ff5c5c", "bg": "rgba(255,92,92,0.08)",   "bar": "#ff5c5c"},
    "fear":     {"emoji": "😨", "color": "#a78bfa", "bg": "rgba(167,139,250,0.08)", "bar": "#a78bfa"},
    "love":     {"emoji": "❤️", "color": "#f472b6", "bg": "rgba(244,114,182,0.08)", "bar": "#f472b6"},
    "surprise": {"emoji": "😲", "color": "#34d399", "bg": "rgba(52,211,153,0.08)",  "bar": "#34d399"},
}


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 40px 0 10px 0;">
    <div class="hero-title">EmoSense</div>
    <div class="hero-sub">Neural Emotion Intelligence</div>
    <div class="accuracy-badge">⚡ 88% Accuracy · Bag of Words · Logistic Regression</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.text_area(
    "YOUR TEXT",
    placeholder="Type anything — a thought, a review, a feeling...",
    height=160,
    label_visibility="visible"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    detect = st.button("Analyse Emotion →", use_container_width=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ── Prediction logic ──────────────────────────────────────────────────────────
if detect:
    if not user_input.strip():
        st.warning("Please type something before analysing.")
    else:
        with st.spinner("Reading between the lines..."):
            time.sleep(0.6)  # slight delay for dramatic effect

        features  = cv.transform([user_input])
        prediction_idx = model.predict(features)[0]

        # manual mapping
        label_map = {
            0: "sadness",
            1: "anger",
            2: "love",
            3: "surprise",
            4: "fear",
            5: "joy"
        }

        prediction = label_map[prediction_idx]
        proba      = model.predict_proba(features)[0]
        confidence = round(max(proba) * 100, 1)
        cfg        = emotion_config.get(prediction, {"emoji": "🤔", "color": "#888", "bg": "#1a1a28", "bar": "#888"})

        # Result card
        st.markdown(f"""
        <div class="result-card" style="background:{cfg['bg']}; border: 1px solid {cfg['color']}30;">
            <span class="result-emoji">{cfg['emoji']}</span>
            <div class="result-label" style="color:{cfg['color']};">{prediction.upper()}</div>
            <div class="result-confidence" style="color:{cfg['color']};">Confidence — {confidence}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Emotion probability bars
        st.markdown('<div class="section-label">Emotion Breakdown</div>', unsafe_allow_html=True)

        classes     = model.classes_
        proba_dict = {
            label_map[int(cls)]: prob
            for cls, prob in zip(classes, proba)
        }
        sorted_proba = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)

        bars_html = ""
        for emotion, prob in sorted_proba:
            ecfg = emotion_config.get(emotion, {"emoji": "🤔", "bar": "#888"})
            pct  = round(prob * 100, 1)
            bars_html += f"""
            <div class="emotion-row">
                <div class="emotion-name">{ecfg['emoji']} {emotion}</div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width:{pct}%; background:{ecfg['bar']};"></div>
                </div>
                <div class="pct-label">{pct}%</div>
            </div>
            """
        st.markdown(bars_html, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    EmoSense · Built with Streamlit · NLP Emotion Detection · 6 Emotions
</div>
""", unsafe_allow_html=True)
