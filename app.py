import os
import pickle
import tempfile

import librosa
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Sound Focus",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APPLICATION
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(99, 102, 241, .12),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(14, 165, 233, .10),
                transparent 30%
            ),
            #f5f7fb;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -.5px;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 42px 45px;
        border-radius: 26px;
        margin-bottom: 30px;
        background:
            linear-gradient(
                135deg,
                #111827 0%,
                #1e1b4b 48%,
                #312e81 100%
            );
        box-shadow: 0 20px 50px rgba(15, 23, 42, .20);
    }

    .hero::before {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        right: -70px;
        top: -90px;
        border-radius: 50%;
        background: rgba(99, 102, 241, .22);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 170px;
        height: 170px;
        right: 170px;
        bottom: -100px;
        border-radius: 50%;
        background: rgba(14, 165, 233, .15);
    }

    .hero-content {
        position: relative;
        z-index: 2;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 14px;
        margin-bottom: 14px;
        border: 1px solid rgba(255, 255, 255, .18);
        border-radius: 999px;
        background: rgba(255, 255, 255, .08);
        color: #dbeafe;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: .4px;
    }

    .hero-title {
        color: white;
        font-size: 42px;
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
    }

    .hero-subtitle {
        color: #dbeafe;
        font-size: 17px;
        margin-top: 12px;
        max-width: 850px;
        line-height: 1.6;
    }

    .hero-mini {
        color: #bfdbfe;
        font-size: 14px;
        margin-top: 16px;
        font-weight: 600;
    }


    /* ========================================================
       SECTION
       ======================================================== */

    .section-title {
        font-size: 23px;
        font-weight: 800;
        color: #172033;
        margin-top: 30px;
        margin-bottom: 18px;
    }


    /* ========================================================
       CARDS
       ======================================================== */

    .card {
        background: rgba(255, 255, 255, .96);
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, .06);
    }

    .result-card {
        padding: 30px;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                #eef2ff,
                #ffffff
            );
        border: 1px solid #c7d2fe;
        box-shadow: 0 12px 35px rgba(79, 70, 229, .10);
    }

    .result-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-value {
        color: #312e81;
        font-size: 34px;
        font-weight: 800;
        margin-top: 5px;
    }

    .result-description {
        color: #475569;
        font-size: 15px;
        line-height: 1.6;
        margin-top: 10px;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        min-height: 115px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, .05);
    }

    .metric-title {
        color: #64748b;
        font-size: 13px;
        font-weight: 700;
    }

    .metric-value {
        color: #172033;
        font-size: 25px;
        font-weight: 800;
        margin-top: 8px;
    }


    /* ========================================================
       PIPELINE
       ======================================================== */

    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        padding: 20px;
        background: white;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 25px rgba(15, 23, 42, .05);
    }

    .pipeline-step {
        padding: 10px 14px;
        border-radius: 12px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 13px;
        font-weight: 700;
    }

    .pipeline-arrow {
        color: #94a3b8;
        font-weight: 900;
    }


    /* ========================================================
       RECOMMENDATION
       ======================================================== */

    .recommendation {
        padding: 25px;
        border-radius: 20px;
        background:
            linear-gradient(
                135deg,
                #ecfeff,
                #eff6ff
            );
        border: 1px solid #bae6fd;
    }

    .recommendation-title {
        color: #0c4a6e;
        font-size: 19px;
        font-weight: 800;
    }

    .recommendation-text {
        color: #334155;
        line-height: 1.7;
        margin-top: 8px;
    }


    /* ========================================================
       INFO BOX
       ======================================================== */

    .info-box {
        padding: 18px 20px;
        border-radius: 16px;
        background: #f8fafc;
        border-left: 5px solid #6366f1;
        color: #475569;
        line-height: 1.6;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #111827 0%,
                #1e1b4b 55%,
                #312e81 100%
            );
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }


    /* ========================================================
       SIDEBAR EXPANDERS
       ======================================================== */

    section[data-testid="stSidebar"] details {
        background: #29264f !important;
        border: 1px solid rgba(255, 255, 255, .14) !important;
        border-radius: 16px !important;
        margin-bottom: 14px !important;
        overflow: hidden !important;
    }

    /* Normal heading */
    section[data-testid="stSidebar"] details summary {
        background: #29264f !important;
        color: #ffffff !important;
        padding: 17px 18px !important;
        font-weight: 800 !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Hover = SAME DARK BACKGROUND */
    section[data-testid="stSidebar"] details summary:hover {
        background: #29264f !important;
        color: #ffffff !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Focus = SAME DARK BACKGROUND */
    section[data-testid="stSidebar"] details summary:focus {
        background: #29264f !important;
        color: #ffffff !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Clicked/Open = SAME DARK BACKGROUND */
    section[data-testid="stSidebar"] details[open] summary {
        background: #29264f !important;
        color: #ffffff !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Heading text */
    section[data-testid="stSidebar"] details summary span,
    section[data-testid="stSidebar"] details summary p,
    section[data-testid="stSidebar"] details summary div {
        color: #ffffff !important;
        background: transparent !important;
    }

    /* Arrow */
    section[data-testid="stSidebar"] details summary svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* Expanded content */
    section[data-testid="stSidebar"] details > div {
        background: #29264f !important;
        color: #f8fafc !important;
        border: none !important;
    }

    /* Details paragraph */
    section[data-testid="stSidebar"] details p {
        background: #29264f !important;
        color: #f8fafc !important;
        padding: 0 18px 12px 18px !important;
        line-height: 1.6;
    }

    /* Caption */
    section[data-testid="stSidebar"] details small {
        background: #29264f !important;
        color: #cbd5e1 !important;
        padding: 0 18px 15px 18px !important;
        display: block;
    }

    /* Remove focus rectangle */
    section[data-testid="stSidebar"] details summary:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }

    /* Prevent nested white backgrounds */
    section[data-testid="stSidebar"] details,
    section[data-testid="stSidebar"] details *,
    section[data-testid="stSidebar"] details summary,
    section[data-testid="stSidebar"] details summary * {
        --background-color: #29264f !important;
    }


    /* ========================================================
       SIDEBAR CUSTOM CARD
       ======================================================== */

    .sidebar-card {
        padding: 18px;
        margin-bottom: 14px;
        border-radius: 16px;
        background: rgba(255, 255, 255, .08);
        border: 1px solid rgba(255, 255, 255, .12);
    }

    .sidebar-title {
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .sidebar-text {
        font-size: 13px;
        color: #cbd5e1 !important;
        line-height: 1.6;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #c7d2fe;
        border-radius: 18px;
        padding: 10px;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 13px;
        padding: 12px 20px;
        background:
            linear-gradient(
                135deg,
                #4f46e5,
                #7c3aed
            );
        color: white;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(79, 70, 229, .25);
    }


    /* ========================================================
       AUDIO
       ======================================================== */

    audio {
        width: 100%;
        border-radius: 12px;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        margin-top: 45px;
        padding-top: 22px;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
        font-size: 13px;
    }


    /* ========================================================
       HIDE STREAMLIT DEFAULT ELEMENTS
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FOR HTML
# ============================================================

def render_html(html):
    """
    Render custom HTML as one continuous block.
    """
    clean_html = " ".join(
        line.strip()
        for line in html.splitlines()
    )
    st.markdown(
        clean_html,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "sound_classifier_50class.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


model = load_model()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True,
    )

    if len(y) == 0:
        raise ValueError(
            "The audio file is empty."
        )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    # RMS
    rms = librosa.feature.rms(
        y=y
    )

    rms_mean = float(
        np.mean(rms)
    )

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y=y
    )

    zcr_mean = float(
        np.mean(zcr)
    )

    # Spectral Centroid
    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
        )
    )

    spectral_centroid_mean = float(
        np.mean(spectral_centroid)
    )

    # Spectral Bandwidth
    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
        )
    )

    spectral_bandwidth_mean = float(
        np.mean(spectral_bandwidth)
    )

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            [
                rms_mean,
                zcr_mean,
                spectral_centroid_mean,
                spectral_bandwidth_mean,
            ],
        ]
    )

    return features, sr


# ============================================================
# SOUND DESCRIPTIONS
# ============================================================

SOUND_DESCRIPTIONS = {

    "airplane":
        "Aircraft or airplane engine sound.",

    "breathing":
        "Human breathing sound.",

    "brushing_teeth":
        "Sound produced while brushing teeth.",

    "can_opening":
        "Sound of opening a metal can.",

    "car_horn":
        "Vehicle horn sound.",

    "cat":
        "Cat vocalization.",

    "chainsaw":
        "Chainsaw operating sound.",

    "chirping_birds":
        "Bird chirping sound.",

    "church_bells":
        "Church bell ringing.",

    "clapping":
        "Hands clapping.",

    "clock_alarm":
        "Alarm clock sound.",

    "clock_tick":
        "Ticking clock sound.",

    "coughing":
        "Human coughing sound.",

    "cow":
        "Cow vocalization.",

    "crackling_fire":
        "Crackling fire sound.",

    "crickets":
        "Cricket insect sounds.",

    "crow":
        "Crow vocalization.",

    "crying_baby":
        "Baby crying sound.",

    "dog":
        "Dog barking or vocalization.",

    "door_wood_creaks":
        "Creaking wooden door sound.",

    "door_wood_knock":
        "Knocking on a wooden door.",

    "drinking_sipping":
        "Drinking or sipping sound.",

    "engine":
        "Engine operating sound.",

    "fireworks":
        "Fireworks sound.",

    "footsteps":
        "Human footsteps.",

    "frog":
        "Frog vocalization.",

    "glass_breaking":
        "Breaking glass sound.",

    "hand_saw":
        "Hand saw cutting sound.",

    "helicopter":
        "Helicopter rotor or engine sound.",

    "hen":
        "Hen vocalization.",

    "insects":
        "General insect sound.",

    "keyboard_typing":
        "Computer keyboard typing sound.",

    "laughing":
        "Human laughter.",

    "mouse_click":
        "Computer mouse clicking sound.",

    "pig":
        "Pig vocalization.",

    "pouring_water":
        "Water being poured.",

    "rain":
        "Rainfall sound.",

    "rooster":
        "Rooster crowing.",

    "sea_waves":
        "Ocean or sea wave sound.",

    "sheep":
        "Sheep vocalization.",

    "siren":
        "Emergency siren sound.",

    "sneezing":
        "Human sneezing sound.",

    "snoring":
        "Human snoring sound.",

    "thunderstorm":
        "Thunderstorm sound.",

    "toilet_flush":
        "Toilet flushing sound.",

    "train":
        "Train movement or engine sound.",

    "vacuum_cleaner":
        "Vacuum cleaner operating sound.",

    "washing_machine":
        "Washing machine operating sound.",

    "water_drops":
        "Individual water drop sounds.",

    "wind":
        "Wind or air movement sound.",
}


# ============================================================
# FOCUS RECOMMENDATION
# ============================================================

LOW_NOISE = {
    "rain",
    "sea_waves",
    "wind",
    "water_drops",
    "crickets",
}

MEDIUM_NOISE = {
    "keyboard_typing",
    "clock_tick",
    "pouring_water",
    "footsteps",
    "breathing",
    "drinking_sipping",
}

HIGH_NOISE = {
    "car_horn",
    "siren",
    "engine",
    "chainsaw",
    "fireworks",
    "thunderstorm",
    "helicopter",
    "train",
    "vacuum_cleaner",
    "washing_machine",
    "glass_breaking",
}


def get_focus_recommendation(sound):

    if sound in LOW_NOISE:
        return (
            "Highly suitable for focused activities. "
            "The detected environment is generally calm "
            "and may support concentration."
        )

    if sound in MEDIUM_NOISE:
        return (
            "Moderately suitable for focus. "
            "The sound may be acceptable for light study "
            "or routine activities."
        )

    if sound in HIGH_NOISE:
        return (
            "Not ideal for deep focus. "
            "The detected environment may be distracting "
            "or relatively noisy."
        )

    return (
        "Focus suitability is moderate. "
        "Consider the surrounding environment before "
        "starting intensive study."
    )


def get_noise_level(sound):

    if sound in LOW_NOISE:
        return "Low"

    if sound in HIGH_NOISE:
        return "High"

    return "Medium"


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero">
        <div class="hero-content">

            <div class="hero-badge">
                🤖 AI + MACHINE LEARNING
            </div>

            <div class="hero-title">
                AI Sound Focus
            </div>

            <div class="hero-subtitle">
                AI-Based Sound Environment Classification
                and Focus Recommendation System
            </div>

            <div class="hero-mini">
                🎧 50-Class Environmental Sound Intelligence
            </div>

        </div>
    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div style="
            font-size:27px;
            font-weight:850;
            margin-bottom:5px;
        ">
            🎧 AI Sound Focus
        </div>

        <div style="
            color:#cbd5e1;
            font-size:13px;
            margin-bottom:25px;
        ">
            Intelligent Audio Analysis
        </div>
        """
    )

    with st.expander(
        "🧠  Machine Learning Model",
        expanded=False
    ):

        st.write(
            "Random Forest Classifier"
        )

        st.caption(
            "The trained model predicts the "
            "environmental sound class from "
            "the extracted audio features."
        )

    with st.expander(
        "🎯  Classification",
        expanded=False
    ):

        st.write(
            "50 ESC-50 environmental sound classes"
        )

        st.caption(
            "The model classifies audio into "
            "one of the 50 classes available "
            "in the ESC-50 dataset."
        )

    with st.expander(
        "📊  Audio Features",
        expanded=False
    ):

        st.write(
            "30 extracted acoustic features"
        )

        st.caption(
            "13 MFCC means + 13 MFCC standard "
            "deviations + RMS + ZCR + spectral "
            "centroid + spectral bandwidth."
        )

    with st.expander(
        "⚙️  AI Pipeline",
        expanded=False
    ):

        st.write(
            "Audio → Features → Classification "
            "→ Environment → Recommendation"
        )

        st.caption(
            "Audio is processed, acoustic features "
            "are extracted, the Random Forest model "
            "predicts the sound, and the system "
            "provides a focus recommendation."
        )


# ============================================================
# MODEL CHECK
# ============================================================

if model is None:

    st.error(
        "Model file 'sound_classifier_50class.pkl' "
        "was not found. Place the trained model "
        "in the same folder as app.py."
    )

    st.stop()


# ============================================================
# PIPELINE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔄 AI Processing Pipeline'
    '</div>',
    unsafe_allow_html=True,
)

render_html(
    """
    <div class="pipeline">

        <div class="pipeline-step">
            🎵 Audio Input
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            ⚙️ Preprocessing
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            📊 Feature Extraction
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            🤖 ML Classification
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            🔊 Sound Environment
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            🎯 Focus Recommendation
        </div>

    </div>
    """
)


# ============================================================
# AUDIO INPUT
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🎵 Audio Input'
    '</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2)


# ============================================================
# UPLOAD
# ============================================================

with left:

    render_html(
        """
        <div class="card">

            <h3>
                📁 Upload Audio
            </h3>

            <p style="color:#64748b;">
                Upload a WAV audio file for
                AI classification.
            </p>

        </div>
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=["wav"],
        label_visibility="collapsed",
    )


# ============================================================
# RECORD
# ============================================================

with right:

    render_html(
        """
        <div class="card">

            <h3>
                🎙️ Record Audio
            </h3>

            <p style="color:#64748b;">
                Record an audio sample directly
                using your microphone.
            </p>

        </div>
        """
    )

    recorded_audio = st.audio_input(
        "Record audio",
        label_visibility="collapsed",
    )


# ============================================================
# SELECT AUDIO
# ============================================================

audio_source = None
audio_name = None


if uploaded_file is not None:

    audio_source = uploaded_file
    audio_name = uploaded_file.name

elif recorded_audio is not None:

    audio_source = recorded_audio
    audio_name = "microphone_recording.wav"


# ============================================================
# AUDIO PREVIEW
# ============================================================

if audio_source is not None:

    st.markdown(
        '<div class="section-title">'
        '🔊 Audio Preview'
        '</div>',
        unsafe_allow_html=True,
    )

    st.audio(
        audio_source,
        format="audio/wav"
    )

    render_html(
        f"""
        <div class="info-box">
            <b>Selected Audio:</b>
            {audio_name}
        </div>
        """
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("")

analyze = st.button(
    "🚀 Analyze Sound",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if audio_source is None:

        st.warning(
            "Please upload or record an "
            "audio file first."
        )

        st.stop()

    temp_path = None

    try:

        with st.spinner(
            "AI is analyzing the audio..."
        ):

            # ------------------------------------------------
            # Save temporary WAV
            # ------------------------------------------------

            audio_bytes = audio_source.getvalue()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav",
            ) as temp_audio:

                temp_audio.write(
                    audio_bytes
                )

                temp_path = temp_audio.name


            # ------------------------------------------------
            # Feature extraction
            # ------------------------------------------------

            features, sr = extract_features(
                temp_path
            )


            # ------------------------------------------------
            # Feature names
            # ------------------------------------------------

            if hasattr(
                model,
                "feature_names_in_"
            ):

                feature_names = list(
                    model.feature_names_in_
                )

            else:

                feature_names = (
                    [
                        f"mfcc_mean_{i + 1}"
                        for i in range(13)
                    ]
                    +
                    [
                        f"mfcc_std_{i + 1}"
                        for i in range(13)
                    ]
                    +
                    [
                        "rms",
                        "zcr",
                        "spectral_centroid",
                        "spectral_bandwidth",
                    ]
                )


            # ------------------------------------------------
            # Feature validation
            # ------------------------------------------------

            if len(feature_names) != len(features):

                raise ValueError(
                    f"Feature mismatch: model expects "
                    f"{len(feature_names)} features, "
                    f"but {len(features)} were extracted."
                )


            # ------------------------------------------------
            # DataFrame
            # ------------------------------------------------

            feature_df = pd.DataFrame(
                [features],
                columns=feature_names,
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = model.predict(
                feature_df
            )[0]


            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            probabilities = model.predict_proba(
                feature_df
            )[0]

            classes = model.classes_


            # ------------------------------------------------
            # Top 10 predictions
            # ------------------------------------------------

            top_indices = np.argsort(
                probabilities
            )[::-1][:10]

            top_classes = classes[
                top_indices
            ]

            top_probabilities = probabilities[
                top_indices
            ]


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            prediction_index = list(
                classes
            ).index(prediction)

            confidence = (
                float(
                    probabilities[
                        prediction_index
                    ]
                ) * 100
            )


        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🎯 AI Classification Result'
            '</div>',
            unsafe_allow_html=True,
        )


        description = SOUND_DESCRIPTIONS.get(
            prediction,
            "Environmental sound detected.",
        )

        noise_level = get_noise_level(
            prediction
        )

        recommendation = (
            get_focus_recommendation(
                prediction
            )
        )


        # ----------------------------------------------------
        # Result Card
        # ----------------------------------------------------

        render_html(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Detected Sound Environment
                </div>

                <div class="result-value">
                    🔊
                    {str(prediction).replace("_", " ").title()}
                </div>

                <div class="result-description">
                    {description}
                </div>

            </div>
            """
        )


        # ====================================================
        # METRICS
        # ====================================================

        st.markdown("")

        m1, m2, m3, m4 = st.columns(4)


        # Confidence
        with m1:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        🎯 Model Confidence
                    </div>

                    <div class="metric-value">
                        {confidence:.2f}%
                    </div>

                </div>
                """
            )


        # Noise
        with m2:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        🔊 Noise Level
                    </div>

                    <div class="metric-value">
                        {noise_level}
                    </div>

                </div>
                """
            )


        # Sample rate
        with m3:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        🎼 Sample Rate
                    </div>

                    <div class="metric-value">
                        {sr:,} Hz
                    </div>

                </div>
                """
            )


        # Features
        with m4:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        📊 Features Used
                    </div>

                    <div class="metric-value">
                        {len(features)}
                    </div>

                </div>
                """
            )


        # ====================================================
        # FOCUS RECOMMENDATION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🧠 Focus Recommendation'
            '</div>',
            unsafe_allow_html=True,
        )


        render_html(
            f"""
            <div class="recommendation">

                <div class="recommendation-title">
                    🎯 Focus Suitability
                </div>

                <div class="recommendation-text">
                    {recommendation}
                </div>

            </div>
            """
        )


        # ====================================================
        # TOP PREDICTIONS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📈 Top AI Predictions'
            '</div>',
            unsafe_allow_html=True,
        )


        probability_df = pd.DataFrame(
            {
                "Sound Class": [
                    str(x)
                    .replace("_", " ")
                    .title()
                    for x in top_classes
                ],

                "Probability (%)": [
                    float(x) * 100
                    for x in top_probabilities
                ],
            }
        )


        chart_df = probability_df.set_index(
            "Sound Class"
        )

        st.bar_chart(
            chart_df
        )


        # ====================================================
        # TOP 10 TABLE
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📋 Prediction Details'
            '</div>',
            unsafe_allow_html=True,
        )


        display_df = probability_df.copy()

        display_df[
            "Probability (%)"
        ] = (
            display_df[
                "Probability (%)"
            ].round(2)
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


        # ====================================================
        # AUDIO FEATURES
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🔬 Extracted Audio Features'
            '</div>',
            unsafe_allow_html=True,
        )


        f1, f2, f3, f4 = st.columns(4)


        # RMS
        with f1:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        RMS Energy
                    </div>

                    <div class="metric-value">
                        {features[26]:.4f}
                    </div>

                </div>
                """
            )


        # ZCR
        with f2:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        Zero Crossing Rate
                    </div>

                    <div class="metric-value">
                        {features[27]:.4f}
                    </div>

                </div>
                """
            )


        # Spectral centroid
        with f3:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        Spectral Centroid
                    </div>

                    <div class="metric-value">
                        {features[28]:.2f}
                    </div>

                </div>
                """
            )


        # Spectral bandwidth
        with f4:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        Spectral Bandwidth
                    </div>

                    <div class="metric-value">
                        {features[29]:.2f}
                    </div>

                </div>
                """
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'ℹ️ About the Prediction'
            '</div>',
            unsafe_allow_html=True,
        )


        render_html(
            """
            <div class="info-box">

                The AI model classifies the uploaded
                audio into one of the 50 environmental
                sound categories from the ESC-50 dataset.

                <br><br>

                The focus recommendation is generated
                using a rule-based interpretation of the
                detected sound environment.

                <br><br>

                <b>Note:</b>
                Model confidence represents the
                classifier's probability estimate and
                does not guarantee that the prediction
                is correct.

            </div>
            """
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        st.error(
            "Unable to analyze the audio."
        )

        st.exception(error)


    # ========================================================
    # CLEAN TEMP FILE
    # ========================================================

    finally:

        if temp_path is not None:

            try:
                os.remove(temp_path)

            except OSError:
                pass


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">

        <b>AI Sound Focus</b>
        &nbsp;•&nbsp;
        AI-Based Sound Environment Classification
        &amp;
        Focus Recommendation System

        <br><br>

        Built with Python • Streamlit • Librosa • Scikit-learn

    </div>
    """
)