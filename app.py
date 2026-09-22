import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# NLTK setup (stopwords only — punkt was downloaded but never used, so it's
# removed to avoid an unnecessary download / potential LookupError on newer
# NLTK versions where the resource path changed).
# ---------------------------------------------------------------------------
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS — polished enterprise-style theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #1e1b4b 0%, #0f172a 55%, #020617 100%);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Hero header */
    .hero-header {
        padding: 18px 26px;
        border-radius: 18px;
        background: linear-gradient(90deg, rgba(79,70,229,0.25) 0%, rgba(56,189,248,0.15) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 22px;
    }
    .hero-header h1 { margin: 0; font-weight: 800; letter-spacing: -0.5px; }
    .hero-header p { margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem; }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 22px 18px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
        transition: transform 0.25s ease, border 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid #6366f1;
    }
    .card-icon { font-size: 1.4rem; margin-bottom: 4px; }
    .card-title {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    .card-value {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 6px;
    }

    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        padding: 22px;
        border-radius: 14px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        margin-top: 15px;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.45);
    }

    /* Buttons */
    .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: opacity 0.2s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover, .stFormSubmitButton>button:hover {
        opacity: 0.88;
        color: white;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .app-footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 40px;
        padding-top: 14px;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
BRAND_COLORS = ["#6366f1", "#38bdf8", "#a855f7", "#0ea5e9", "#818cf8", "#c084fc", "#22d3ee"]


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter",
        font_color="#f8fafc",
        template=PLOTLY_TEMPLATE,
        legend_title_text="",
    )
    return fig


# ---------------------------------------------------------------------------
# Load Models — errors are now surfaced instead of silently swallowed
# ---------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("resume_model.pkl", "rb"))
        tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
        encoder = pickle.load(open("label_encoder.pkl", "rb"))
        return model, tfidf, encoder, None
    except FileNotFoundError as e:
        return None, None, None, f"Model file missing: {e.filename}"
    except Exception as e:
        return None, None, None, f"Could not load model assets: {e}"


model, tfidf, encoder, load_error = load_assets()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def hero(title, subtitle):
    st.markdown(f"""
    <div class="hero-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(col, icon, title, value):
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="card-icon">{icon}</div>'
            f'<div class="card-title">{title}</div>'
            f'<div class="card-value">{value}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=90)
st.sidebar.title("AI Screening Suite")
st.sidebar.caption("v2.0 · Enterprise Edition")
menu = st.sidebar.radio("Navigate", ["Dashboard Analytics", "Single Resume Predictor", "Bulk Resume Screening"])
st.sidebar.write("---")
st.sidebar.caption("Powered by TF-IDF + Random Forest")

# ---------------------------------------------------------------------------
# Page 1: Dashboard Analytics
# ---------------------------------------------------------------------------
if menu == "Dashboard Analytics":
    hero("📊 Candidate Data Insights & Analytics",
         "Visual overview of candidates, skills distribution, and AI screening scores.")

    uploaded_file = st.file_uploader("Upload Data CSV for Live Dashboard", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the uploaded CSV: {e}")
            st.stop()
    else:
        try:
            df = pd.read_csv("AI_Resume_Screening (1).csv")
        except Exception:
            st.warning("Please upload 'AI_Resume_Screening (1).csv' to view analytics.")
            st.stop()

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, "👥", "Total Candidates", len(df))

    if 'Experience (Years)' in df.columns:
        avg_exp = f"{round(df['Experience (Years)'].mean(), 1)} Yrs"
    else:
        avg_exp = "N/A"
    metric_card(m2, "📈", "Avg Experience", avg_exp)

    if 'AI Score (0-100)' in df.columns:
        avg_score = f"{round(df['AI Score (0-100)'].mean(), 1)}/100"
    else:
        avg_score = "N/A"
    metric_card(m3, "🎯", "Avg AI Score", avg_score)

    if 'Recruiter Decision' in df.columns:
        hired = len(df[df['Recruiter Decision'] == 'Hire'])
    else:
        hired = "N/A"
    metric_card(m4, "✅", "Hired Count", hired)

    st.write("")
    st.write("---")

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        if 'Job Role' in df.columns:
            fig1 = px.pie(df, names='Job Role', title='Job Role Distribution', hole=0.45,
                          color_discrete_sequence=BRAND_COLORS)
            st.plotly_chart(style_fig(fig1), use_container_width=True)
        else:
            st.info("Add a 'Job Role' column to see the role distribution chart.")

    with col2:
        if 'Experience (Years)' in df.columns and 'AI Score (0-100)' in df.columns:
            plot_df = df.copy()
            if 'Projects Count' in plot_df.columns:
                plot_df['_size'] = plot_df['Projects Count'].clip(lower=0).fillna(0)
                size_kwarg = {"size": "_size", "size_max": 22}
            else:
                size_kwarg = {}

            color_kwarg = {}
            if 'Recruiter Decision' in plot_df.columns:
                color_kwarg = {
                    "color": "Recruiter Decision",
                    "color_discrete_map": {'Hire': '#22c55e', 'Reject': '#ef4444'}
                }

            fig2 = px.scatter(
                plot_df, x='Experience (Years)', y='AI Score (0-100)',
                title='Experience vs AI Score', **color_kwarg, **size_kwarg
            )
            st.plotly_chart(style_fig(fig2), use_container_width=True)
        else:
            st.info("Add 'Experience (Years)' and 'AI Score (0-100)' columns to see this chart.")

# ---------------------------------------------------------------------------
# Page 2: Single Resume Predictor
# ---------------------------------------------------------------------------
elif menu == "Single Resume Predictor":
    hero("🎯 AI Job Role Predictor",
         "Enter candidate skills to predict the appropriate job role using the trained model.")

    if model is None:
        st.error(load_error or "Model files not found! Please ensure 'resume_model.pkl', "
                                "'tfidf_vectorizer.pkl', and 'label_encoder.pkl' are present.")
    else:
        with st.form("resume_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Candidate Name", "John Doe")
                exp = st.number_input("Experience (Years)", min_value=0, max_value=40, value=3)
            with col2:
                edu = st.selectbox("Education", ["B.Sc", "B.Tech", "M.Tech", "MBA", "PhD"])
                projects = st.number_input("Projects Count", min_value=0, max_value=30, value=5)

            skills_input = st.text_area(
                "Candidate Skills (Comma Separated)",
                "Python, Machine Learning, Deep Learning, SQL, TensorFlow"
            )
            st.caption("Note: the current model predicts role from skills text only "
                       "(experience/education/projects are captured for your records).")

            submit_btn = st.form_submit_button("Screen & Predict Candidate")

        if submit_btn:
            if not skills_input.strip():
                st.warning("Please enter at least one skill.")
            else:
                try:
                    cleaned = clean_text(skills_input)
                    vectorized = tfidf.transform([cleaned])
                    pred_code = model.predict(vectorized)[0]
                    pred_role = encoder.inverse_transform([pred_code])[0]
                    probs = model.predict_proba(vectorized)[0]
                    confidence = round(np.max(probs) * 100, 2)

                    st.markdown(
                        f'<div class="prediction-box">Predicted Job Role: {pred_role} '
                        f'<br><span style="font-size:1rem; opacity:0.85;">Confidence: {confidence}%</span></div>',
                        unsafe_allow_html=True
                    )

                    st.write("### Prediction Confidence Breakdown")
                    class_names = encoder.classes_
                    order = np.argsort(probs)
                    fig = go.Figure(go.Bar(
                        x=(probs * 100)[order],
                        y=np.array(class_names)[order],
                        orientation='h',
                        marker=dict(color='#6366f1')
                    ))
                    fig.update_layout(xaxis_title="Probability (%)", yaxis_title="Job Role")
                    st.plotly_chart(style_fig(fig), use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction failed — the model may expect different input features. Details: {e}")

# ---------------------------------------------------------------------------
# Page 3: Bulk Resume Screening
# ---------------------------------------------------------------------------
elif menu == "Bulk Resume Screening":
    hero("⚡ Bulk Resume Automated Screening",
         "Upload a candidate dataset to automatically classify roles and generate prediction scores.")

    uploaded_batch = st.file_uploader("Upload CSV File for Bulk Screening", type=["csv"], key="bulk")

    if model is None:
        st.error(load_error or "Model files not found! Please ensure the model files are present.")
    elif uploaded_batch is not None:
        try:
            batch_df = pd.read_csv(uploaded_batch)
        except Exception as e:
            st.error(f"Could not read the uploaded CSV: {e}")
            st.stop()

        if "Skills" not in batch_df.columns:
            st.error("The uploaded CSV must contain a 'Skills' column.")
        else:
            try:
                with st.spinner("Processing Resumes..."):
                    cleaned_skills = batch_df["Skills"].fillna("").apply(clean_text)
                    vectorized_skills = tfidf.transform(cleaned_skills)
                    preds = model.predict(vectorized_skills)
                    probs_batch = model.predict_proba(vectorized_skills)

                    batch_df["AI Predicted Role"] = encoder.inverse_transform(preds)
                    batch_df["AI Confidence (%)"] = np.round(np.max(probs_batch, axis=1) * 100, 2)

                st.success(f"Batch Processing Complete! {len(batch_df)} candidates screened.")

                display_cols = [c for c in ["Name", "Skills"] if c in batch_df.columns]
                display_cols += ["AI Predicted Role", "AI Confidence (%)"]
                st.dataframe(batch_df[display_cols].head(20), use_container_width=True)

                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Classified Candidates Data",
                    data=csv,
                    file_name="screened_candidates.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Bulk prediction failed — the model may expect different input features. Details: {e}")
    else:
        st.info("Upload a CSV with a 'Skills' column to begin bulk screening.")

# ---------------------------------------------------------------------------
st.markdown('<div class="app-footer">© 2026 AI Resume Screening System · Internal Use</div>',
            unsafe_allow_html=True)
