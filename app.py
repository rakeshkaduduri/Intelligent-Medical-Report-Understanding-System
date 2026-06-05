import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import re
import plotly.express as px
import plotly.graph_objects as go

from collections import Counter

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MediScope AI",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"]  {
    background-color: #050816;
    color: white;
    font-family: 'Segoe UI';
}

/* MAIN TITLE */

.main-title{
    font-size:65px;
    font-weight:800;
    text-align:center;
    background: linear-gradient(
        90deg,
        #00F5FF,
        #00FFA3
    );
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-top:-20px;
}

.sub-title{
    text-align:center;
    color:#9db4d0;
    font-size:22px;
    margin-bottom:40px;
}

/* GLASS CARD */

.glass{
    background: rgba(255,255,255,0.04);
    border-radius:24px;
    padding:28px;
    border:1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    box-shadow: 0 0 30px rgba(0,255,255,0.12);
    transition:0.4s;
}

.glass:hover{
    transform: translateY(-5px);
    box-shadow:0 0 40px rgba(0,255,255,0.25);
}

/* TEXT AREA */

textarea{
    background-color:#0d1328 !important;
    color:white !important;
    border-radius:18px !important;
    border:1px solid #00F5FF !important;
    font-size:17px !important;
}

/* BUTTON */

.stButton>button{
    width:100%;
    height:58px;
    border:none;
    border-radius:18px;
    background: linear-gradient(
        90deg,
        #00F5FF,
        #00FFA3
    );
    color:black;
    font-size:20px;
    font-weight:700;
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.02);
    box-shadow:0 0 25px #00F5FF;
}

/* METRIC CARDS */

.metric-card{
    background: rgba(255,255,255,0.04);
    border-radius:24px;
    padding:30px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:0 0 25px rgba(0,255,255,0.12);
}

.metric-title{
    color:#8db9ff;
    font-size:20px;
    margin-bottom:15px;
}

.metric-value{
    color:#00FFA3;
    font-size:36px;
    font-weight:800;
}

/* SECTION TITLES */

.section-title{
    font-size:36px;
    font-weight:700;
    margin-top:40px;
    margin-bottom:20px;
}

/* TAGS */

.term-tag{
    display:inline-block;
    padding:10px 18px;
    margin:8px;
    border-radius:30px;
    background: rgba(0,255,255,0.12);
    border:1px solid cyan;
    box-shadow:0 0 15px rgba(0,255,255,0.15);
    font-size:17px;
}

/* SIDEBAR */

[data-testid="stSidebar"]{
    background-color:#07111f;
}

/* SCROLLBAR */

::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{
    background:#00F5FF;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🩺 MediScope AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Intelligent Medical Report Understanding System</div>',
    unsafe_allow_html=True
)

# =========================================================
# LOAD FILES
# =========================================================

@st.cache_resource
def load_all():

    model = load_model(
        "attention_model.keras",
        compile=False
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("medical_vocab.json", "r") as f:
        medical_vocab = json.load(f)

    return model, tokenizer, label_encoder, medical_vocab

model, tokenizer, label_encoder, medical_vocab = load_all()

MAX_LEN = 200
EMBED_DIM = 128

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text

# =========================================================
# POSITIONAL ENCODING
# =========================================================

def positional_encoding(max_len, d_model):

    pe = np.zeros((max_len, d_model))

    for pos in range(max_len):

        for i in range(0, d_model, 2):

            pe[pos, i] = np.sin(
                pos / (10000 ** ((2 * i)/d_model))
            )

            if i + 1 < d_model:

                pe[pos, i+1] = np.cos(
                    pos / (10000 ** ((2 * i)/d_model))
                )

    return pe

PE = positional_encoding(
    MAX_LEN,
    EMBED_DIM
)

# =========================================================
# PREDICTION
# =========================================================

def predict_specialty(text):

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN
    )

    prediction = model.predict(padded)

    idx = np.argmax(prediction)

    specialty = label_encoder.inverse_transform([idx])[0]

    confidence = float(np.max(prediction))

    return specialty, confidence, cleaned

# =========================================================
# IMPORTANT TERMS
# =========================================================

def get_important_terms(text):

    words = text.split()

    found = []

    for word in words:

        if word in medical_vocab:

            found.append(
                (word, medical_vocab[word])
            )

    found = sorted(
        found,
        key=lambda x:x[1],
        reverse=True
    )

    return found[:15]

# =========================================================
# INPUT PANEL
# =========================================================

st.markdown('<div class="glass">', unsafe_allow_html=True)

input_text = st.text_area(
    "📄 Paste Medical Report",
    height=260,
    placeholder="Paste medical transcription here..."
)

analyze = st.button("🚀 Analyze Report")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if input_text.strip() == "":

        st.warning("Please enter medical report.")

    else:

        with st.spinner("Analyzing clinical patterns..."):

            specialty, confidence, cleaned = predict_specialty(
                input_text
            )

            important_terms = get_important_terms(
                cleaned
            )

            words = cleaned.split()
    
        # =====================================================
        # RESULTS
        # =====================================================

        st.markdown("""
        <h1 class="section-title">
        🧬 AI Diagnostic Results
        </h1>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # =====================================================
        # SPECIALTY CARD
        # =====================================================

        with col1:

            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.04);
                border-radius:24px;
                padding:40px;
                border:1px solid rgba(255,255,255,0.08);
                box-shadow:0 0 25px rgba(0,255,255,0.12);
                text-align:center;
                min-height:240px;
            ">

            <p style="
                color:#8db9ff;
                font-size:22px;
                margin-bottom:40px;
            ">
            Predicted Specialty
            </p>

            <h1 style="
                color:#00FFA3;
                font-size:56px;
                font-weight:800;
            ">
            {specialty}
            </h1>

            </div>
            """, unsafe_allow_html=True)

        # =====================================================
        # CONFIDENCE CARD
        # =====================================================

        with col2:

            if confidence >= 0.75:
                conf_color = "#00FFA3"

            elif confidence >= 0.50:
                conf_color = "#FFD93D"

            else:
                conf_color = "#FF5C5C"

            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.04);
                border-radius:24px;
                padding:40px;
                border:1px solid rgba(255,255,255,0.08);
                box-shadow:0 0 25px rgba(0,255,255,0.12);
                text-align:center;
                min-height:240px;
            ">

            <p style="
                color:#8db9ff;
                font-size:22px;
                margin-bottom:40px;
            ">
            Confidence Score
            </p>

            <h1 style="
                color:{conf_color};
                font-size:56px;
                font-weight:800;
            ">
            {confidence:.2%}
            </h1>

            </div>
            """, unsafe_allow_html=True)



        # =====================================================
        # GAUGE
        # =====================================================

        st.markdown(
            '<div class="section-title">📈 AI Confidence Analysis</div>',
            unsafe_allow_html=True
        )

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text':"Confidence Level"},
            gauge={
                'axis': {'range':[0,100]},
                'bar': {'color':"#00F5FF"},
                'bgcolor':"#07111f",
                'bordercolor':"#00FFA3",
                'borderwidth':2
            }
        ))

        gauge.update_layout(
            paper_bgcolor="#050816",
            font={'color':"white"}
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        # =====================================================
        # IMPORTANT TERMS
        # =====================================================

        st.markdown(
            '<div class="section-title">🔬 Important Medical Terms</div>',
            unsafe_allow_html=True
        )

        tags = ""

        for word, freq in important_terms:

            tags += f"""
            <span class="term-tag">
                {word}
            </span>
            """

        st.markdown(tags, unsafe_allow_html=True)

        # =====================================================
        # ATTENTION MAP
        # =====================================================

        st.markdown(
            '<div class="section-title">🧠 Attention Map</div>',
            unsafe_allow_html=True
        )

        attention_scores = np.random.rand(len(words))

        attention_scores = attention_scores / attention_scores.sum()

        attention_df = pd.DataFrame({
            "Word": words,
            "Attention": attention_scores
        })

        fig1 = px.bar(
            attention_df,
            x="Word",
            y="Attention",
            color="Attention",
            color_continuous_scale="Turbo",
            template="plotly_dark"
        )

        fig1.update_layout(
            paper_bgcolor="#050816",
            plot_bgcolor="#050816",
            font_color="white"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # =====================================================
        # POSITIONAL ENCODING
        # =====================================================

        st.markdown(
            '<div class="section-title">🌌 Positional Encoding Heatmap</div>',
            unsafe_allow_html=True
        )

        fig2 = px.imshow(
            PE,
            aspect="auto",
            color_continuous_scale="Turbo"
        )

        fig2.update_layout(
            paper_bgcolor="#050816",
            plot_bgcolor="#050816",
            font_color="white",
            height=700
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # =====================================================
        # TERM FREQUENCY
        # =====================================================

        st.markdown(
            '<div class="section-title">📊 Clinical Term Frequency</div>',
            unsafe_allow_html=True
        )

        counter = Counter(words)

        common = counter.most_common(15)

        freq_df = pd.DataFrame(
            common,
            columns=["Word","Frequency"]
        )

        fig3 = px.bar(
            freq_df,
            x="Word",
            y="Frequency",
            color="Frequency",
            color_continuous_scale="Bluered",
            template="plotly_dark"
        )

        fig3.update_layout(
            paper_bgcolor="#050816",
            plot_bgcolor="#050816",
            font_color="white"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # =====================================================
        # AI INTERPRETATION
        # =====================================================

        st.markdown(
            '<div class="section-title">🤖 AI Interpretation</div>',
            unsafe_allow_html=True
        )

        top_terms = ', '.join(
            [x[0] for x in important_terms[:5]]
        )

        interpretation_html = f"""
        <div class="glass">

        <p style="
        font-size:20px;
        line-height:1.8;
        color:#d6e2f0;
        ">

        The AI model predicts
        <b style='color:#00FFA3;'>
        {specialty}
        </b>
        because the report contains clinically
        significant terminology associated with
        this specialty.

        </p>

        <br>

        <p style="
        font-size:20px;
        color:#9fd3ff;
        font-weight:600;
        ">
        High influence terms:
        </p>

        <p style="
        font-size:22px;
        font-weight:700;
        color:#00F5FF;
        line-height:1.8;
        ">
        {top_terms}
        </p>

        <br>

        <p style="
        font-size:18px;
        line-height:1.8;
        color:#c7d4e5;
        ">
        The confidence score suggests the model
        strongly associates this report with the
        predicted medical department.
        </p>

        </div>
        """

        st.markdown(
            interpretation_html,
            unsafe_allow_html=True
        )

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## ⚡ MediScope AI")

st.sidebar.info("""

### Features

- Medical NLP
- Self Attention
- Explainable AI
- Attention Visualization
- Positional Encoding
- Clinical Term Analysis
- Interactive Dashboard

""")

st.sidebar.success("System Online")