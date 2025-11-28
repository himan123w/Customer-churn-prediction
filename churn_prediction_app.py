
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import Dict, Any
import io
import base64

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Theme handling (CSS-based light/dark)
# -------------------------
def inject_theme(theme: str = "Auto"):
    # Keep colors subtle to avoid fighting with Streamlit's own theme.
    if theme == "Dark":
        css = """
        <style>
          :root {
            --bg: #0f172a;
            --panel: #111827;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --border: rgba(255,255,255,0.08);
            --accentbg: #0b1225;
            --accentborder: rgba(64,120,255,0.35);
          }
          .stApp { background: var(--bg); color: var(--text); }
          .metric-card {
              background: var(--panel) !important;
              border: 1px solid var(--border);
              box-shadow: 0 2px 14px rgba(0,0,0,0.35);
          }
          .accent {
              background: linear-gradient(135deg, #0b1225 0%, #0f172a 100%) !important;
              border: 1px solid var(--accentborder) !important;
          }
          .small { color: var(--muted) !important; }
          .callout {
              border-left: 4px solid #7096ff;
              background: #0b1225;
              color: var(--text);
          }
        </style>
        """
    elif theme == "Light":
        css = """
        <style>
          :root {
            --bg: #ffffff;
            --panel: #ffffff;
            --text: #111827;
            --muted: #6b7280;
            --border: rgba(49,51,63,0.15);
            --accentbg: #ecf2ff;
            --accentborder: rgba(64,120,255,0.25);
          }
          .stApp { background: var(--bg); color: var(--text); }
          .metric-card {
              background: var(--panel) !important;
              border: 1px solid var(--border);
              box-shadow: 0 2px 12px rgba(0,0,0,0.06);
          }
          .accent {
              background: linear-gradient(135deg, #ecf2ff 0%, #f8fbff 100%) !important;
              border: 1px solid var(--accentborder) !important;
          }
          .small { color: var(--muted) !important; }
          .callout {
              border-left: 4px solid #4c78ff;
              background: #f6f8ff;
              color: var(--text);
          }
        </style>
        """
    else:
        css = ""  # Let Streamlit decide
    st.markdown(css, unsafe_allow_html=True)

if "theme" not in st.session_state:
    st.session_state["theme"] = "Auto"

# -------------------------
# Base styling
# -------------------------
st.markdown(
    """
    <style>
      .main > div {padding-top: 1rem;}
      .chip {
          display: inline-block;
          padding: 6px 12px;
          border-radius: 999px;
          font-weight: 600;
      }
      .chip-ok {
          background: #e9f9ee;
          color: #0f8b3e;
          border: 1px solid #c2efd1;
      }
      .chip-warn {
          background: #fff2f0;
          color: #b42411;
          border: 1px solid #ffd1cc;
      }
      .tight { margin-top: -10px; }
      .metric-card { border-radius: 16px; padding: 18px 18px 6px 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Paths & loading
# -------------------------
HERE = Path(__file__).resolve().parent
PIPELINE_PATH = HERE / "churn_pipeline.pkl"
UI_META_PATH  = HERE / "churn_ui_meta.json"

@st.cache_resource
def load_artifacts():
    pipe = joblib.load(PIPELINE_PATH)
    with open(UI_META_PATH) as f:
        ui_meta = json.load(f)
    return pipe, ui_meta

try:
    pipe, ui_meta = load_artifacts()
except Exception as e:
    st.error("Could not load the trained pipeline or UI metadata. "
             "Place 'churn_pipeline.pkl' and 'churn_ui_meta.json' in the same folder as this app.")
    st.exception(e)
    st.stop()

# -------------------------
# Sidebar — Controls (Theme + Logo + Inputs)
# -------------------------
with st.sidebar:
    st.header("Appearance")
    st.session_state["theme"] = st.selectbox("Theme", ["Auto", "Light", "Dark"], index=["Auto","Light","Dark"].index(st.session_state["theme"]))
    inject_theme(st.session_state["theme"])

    st.markdown("---")
    st.header("Branding")
    logo_file = st.file_uploader("Upload logo (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=False)
    logo_url = st.text_input("...or paste logo URL")
    st.caption("Logo appears in the header if provided.")

    st.markdown("---")
    st.header("Input Features")  # (unchanged feature set)
    n_defaults = ui_meta["defaults"]

    tenure  = st.number_input("tenure (months)", 0, 1000, int(n_defaults.get("tenure", 12)))
    monthly = st.number_input("MonthlyCharges", 0.0, 2000.0, float(n_defaults.get("MonthlyCharges", 70.0)), step=0.1)
    total   = st.number_input("TotalCharges", 0.0, 100000.0, float(n_defaults.get("TotalCharges", 1500.0)), step=0.1)
    senior  = st.selectbox("SeniorCitizen (0=No, 1=Yes)", [0, 1], index=[0,1].index(int(n_defaults.get("SeniorCitizen", 0))))

    cat_values: Dict[str, Any] = {}
    for col in ui_meta["categorical_cols"]:
        opts = ui_meta["options"].get(col, [])
        if not opts:
            opts = ["N/A"]
        cat_values[col] = st.selectbox(col, options=opts)

    st.markdown("---")
    go = st.button("🔮 Predict", use_container_width=True)

# -------------------------
# Header (with optional logo)
# -------------------------
def render_logo(logo_file, logo_url):
    if logo_file is not None:
        data = logo_file.read()
        b64 = base64.b64encode(data).decode("utf-8")
        return f'<img src="data:image/png;base64,{b64}" style="height:42px; margin-right:10px;" />'
    if logo_url:
        return f'<img src="{logo_url}" style="height:42px; margin-right:10px;" />'
    return ""

left, right = st.columns([0.75, 0.25])
with left:
    logo_html = render_logo(logo_file, logo_url)
    st.markdown(f"{logo_html}<span style='font-size:28px; font-weight:700;'>Telco Customer Churn Predictor</span>",
                unsafe_allow_html=True)
    st.markdown(
        "Predict customer churn using a **RandomForest** model with scaling + one-hot encoding."
    )
with right:
    with st.container():
        st.markdown('<div class="metric-card accent">', unsafe_allow_html=True)
        st.metric("Model Accuracy (test)", f"{ui_meta['metrics']['accuracy']:.3f}")
        st.markdown('<p class="small tight">Held-out split</p></div>', unsafe_allow_html=True)

st.markdown("")

# -------------------------
# Tabs: Single Prediction | Batch Scoring | About
# -------------------------
tab_pred, tab_batch, tab_about = st.tabs(["🔍 Single Prediction", "📂 Batch Scoring", "ℹ️ About"])

with tab_pred:
    pred_col, expl_col = st.columns([0.62, 0.38])

    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "SeniorCitizen": senior,
    }
    data.update(cat_values)
    X_new = pd.DataFrame([data])

    with pred_col:
        st.subheader("Result")
        if go:
            pred = pipe.predict(X_new)[0]
            proba = pipe.predict_proba(X_new)[0][1] if hasattr(pipe, "predict_proba") else None

            is_churn = not (str(pred) in ["0", "No"])
            if is_churn:
                st.markdown('<span class="chip chip-warn">⚠️ Likely to CHURN</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="chip chip-ok">✅ Will NOT churn</span>', unsafe_allow_html=True)

            if proba is not None:
                st.markdown("")
                st.markdown("**Churn Probability**")
                st.progress(min(max(float(proba), 0.0), 1.0))
                st.caption(f"{proba:.2%}")

            with st.expander("🔎 View request snapshot"):
                st.dataframe(X_new)
        else:
            st.info("Set inputs in the sidebar and click **Predict**.")

    with expl_col:
        st.subheader("Quick Stats")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Tenure (months)", tenure)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Monthly Charges", f"{monthly:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Charges", f"{total:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Senior Citizen", "Yes" if int(senior)==1 else "No")
            st.markdown('</div>', unsafe_allow_html=True)

with tab_batch:
    st.subheader("Batch CSV Scoring")
    st.caption("Upload a CSV with the same columns used for training (all feature columns from the Telco dataset, excluding `customerID` and `Churn`).")
    csv_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
    if csv_file is not None:
        try:
            df_in = pd.read_csv(csv_file)
            # Basic validation: ensure required columns exist
            required_cols = ui_meta["numeric_cols"] + ui_meta["categorical_cols"]
            missing = [c for c in required_cols if c not in df_in.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                # Predict
                probs = pipe.predict_proba(df_in)[:, 1] if hasattr(pipe, "predict_proba") else None
                preds = pipe.predict(df_in)
                out = df_in.copy()
                out["prediction"] = preds
                if probs is not None:
                    out["churn_probability"] = probs

                st.success(f"Scored {len(out)} rows.")
                st.dataframe(out.head(50))

                # Download button
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download predictions CSV",
                    data=csv_bytes,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error("Could not read or score the CSV. Please ensure the schema matches the training features.")
            st.exception(e)
    else:
        st.info("Upload a CSV to begin.")

with tab_about:
    st.write(
        """
        **Pipeline**: ColumnTransformer → (StandardScaler for numeric, OneHotEncoder for categorical) → RandomForestClassifier  
        **Inputs**: Identical to training features.  
        **Tip**: Use *Batch Scoring* tab for many customers.
        """
    )
