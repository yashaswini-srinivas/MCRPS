import os
import random
import string

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from orchestrator.agentic_pipeline import run_agentic_mcrps

load_dotenv()  # Automatically loads variables from .env

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Diabetic Complication Risk System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — dark card style, compact sidebar inputs, coloured risk bars
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #0e1117; border-right: 1px solid #1e2530; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] .sidebar-header {
        font-size: 1.05rem; font-weight: 700;
        color: #ffffff !important; margin-bottom: 0.25rem;
    }
    [data-testid="stSidebar"] label { font-size: 0.72rem !important; color: #aaaaaa !important; }

    /* ── Cards ── */
    .card {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1rem; font-weight: 700; color: #e6edf3;
        margin-bottom: 0.9rem; display: flex; align-items: center; gap: 0.4rem;
    }

    /* ── Risk bar rows ── */
    .risk-row { display: flex; align-items: center; margin-bottom: 0.65rem; }
    .risk-label { width: 140px; font-size: 0.82rem; color: #c9d1d9; flex-shrink: 0; }
    .risk-bar-bg {
        flex: 1; background: #21262d; border-radius: 6px;
        height: 8px; overflow: hidden; margin: 0 0.6rem;
    }
    .risk-bar-fill { height: 100%; border-radius: 6px; }
    .risk-value { width: 52px; text-align: right; font-size: 0.82rem;
                  font-weight: 600; flex-shrink: 0; }

    /* ── Risk level badge ── */
    .badge {
        display: inline-block; padding: 0.18rem 0.7rem;
        border-radius: 20px; font-size: 0.75rem; font-weight: 700;
    }
    .badge-high    { background: #3d1f1f; color: #f87171; border: 1px solid #f87171; }
    .badge-moderate{ background: #3d2f12; color: #fbbf24; border: 1px solid #fbbf24; }
    .badge-low     { background: #143022; color: #34d399; border: 1px solid #34d399; }
    .badge-unknown { background: #1f2533; color: #94a3b8; border: 1px solid #94a3b8; }

    /* ── Trace / log box ── */
    .trace-box {
        background: #0d1117; border: 1px solid #21262d;
        border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.78rem;
        color: #7ee787; font-family: 'Courier New', monospace;
        max-height: 220px; overflow-y: auto;
    }
    .trace-line { margin: 0.18rem 0; }

    /* ── Image caption ── */
    .img-caption {
        text-align: center; font-size: 0.72rem; color: #8b949e;
        margin-top: 0.3rem;
    }

    /* ── Divider ── */
    hr { border-color: #21262d !important; }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _risk_bar_color(value: float) -> str:
    """Green → amber → red gradient colour for the risk bar."""
    if value >= 0.55:
        return "#f87171"   # red
    if value >= 0.35:
        return "#fbbf24"   # amber
    return "#34d399"       # green


def _badge_class(risk_level: str) -> str:
    mapping = {
        "HIGH": "badge-high",
        "MODERATE": "badge-moderate",
        "LOW": "badge-low",
    }
    return mapping.get(str(risk_level).upper(), "badge-unknown")


def _risk_row_html(label: str, value: float) -> str:
    pct = min(100.0, value * 100)
    color = _risk_bar_color(value)
    return (
        f'<div class="risk-row">'
        f'  <span class="risk-label">{label}</span>'
        f'  <div class="risk-bar-bg">'
        f'    <div class="risk-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        f'  </div>'
        f'  <span class="risk-value" style="color:{color};">{pct:.1f}%</span>'
        f'</div>'
    )


def _gauge_figure(dcrs: float) -> go.Figure:
    """Plotly semi-circle gauge matching the reference design."""
    pct = dcrs * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": "#e6edf3"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#444c56",
                "tickfont": {"color": "#8b949e", "size": 10},
                "dtick": 20,
            },
            "bar": {"color": "#34d399", "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "#143022"},
                {"range": [35, 55], "color": "#3d2f12"},
                {"range": [55, 100], "color": "#3d1f1f"},
            ],
            "threshold": {
                "line": {"color": "#e6edf3", "width": 2},
                "thickness": 0.75,
                "value": pct,
            },
        },
        title={"text": "Composite Risk Score (DCRS)",
               "font": {"size": 13, "color": "#8b949e"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin={"t": 40, "b": 0, "l": 20, "r": 20},
        font={"color": "#e6edf3"},
    )
    return fig


def _random_patient_id() -> str:
    return "P_" + "".join(random.choices(string.digits, k=5))


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = _random_patient_id()
if "pipeline_output" not in st.session_state:
    st.session_state["pipeline_output"] = None


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Patient Clinical Profile
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p class="sidebar-header">🩺&nbsp; Patient Clinical Profile</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── LLM Provider ──
    st.markdown('<span style="font-size:0.72rem;color:#aaa;">🤖 LLM Provider</span>',
                unsafe_allow_html=True)
    llm_provider = st.selectbox(
        "LLM Provider", ["Auto-Detect", "Groq", "Gemini", "OpenAI", "Anthropic"],
        label_visibility="collapsed",
    )

    # ── API Key ──
    st.markdown(
        '<span style="font-size:0.72rem;color:#aaa;">🔑 API Key (Groq / Gemini / Anthropic / OpenAI)</span>',
        unsafe_allow_html=True,
    )
    api_key = st.text_input("API Key", type="password", label_visibility="collapsed",
                            placeholder="Paste your API key…")

    # ── Key-prefix auto-detection ─────────────────────────────────────────────
    def _route_api_key(key: str, provider_override: str) -> str:
        """
        Set only the correct provider env-var; clear all others.
        Returns a display label for the detected provider.
        """
        if not key:
            return ""

        for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
            os.environ.pop(var, None)

        if provider_override == "Anthropic":
            os.environ["ANTHROPIC_API_KEY"] = key
            return "Anthropic Claude"
        if provider_override == "Gemini":
            os.environ["GEMINI_API_KEY"] = key
            return "Google Gemini"
        if provider_override == "OpenAI":
            os.environ["OPENAI_API_KEY"] = key
            return "OpenAI GPT"
        if provider_override == "Groq":
            os.environ["GROQ_API_KEY"] = key
            return "Groq"

        if key.startswith("gsk_"):
            os.environ["GROQ_API_KEY"] = key
            return "Groq"
        if key.startswith("sk-ant-"):
            os.environ["ANTHROPIC_API_KEY"] = key
            return "Anthropic Claude"
        if key.startswith("AIza") or key.startswith("AI"):
            os.environ["GEMINI_API_KEY"] = key
            return "Google Gemini"
        if key.startswith("sk-"):
            os.environ["OPENAI_API_KEY"] = key
            return "OpenAI GPT"

        os.environ["GROQ_API_KEY"] = key
        return "Groq (assumed)"

    detected_provider = _route_api_key(api_key, llm_provider)

    if api_key and detected_provider:
        _prov_colors = {
            "Anthropic Claude": ("#2d1a3d", "#c084fc"),
            "Google Gemini":    ("#1a2d1a", "#4ade80"),
            "OpenAI GPT":       ("#1a2535", "#60a5fa"),
            "Groq":             ("#2d2010", "#fb923c"),
            "Groq (assumed)":   ("#2d2010", "#fb923c"),
        }
        bg, fg = _prov_colors.get(detected_provider, ("#1e1e1e", "#94a3b8"))
        st.markdown(
            f'<div style="margin-top:0.3rem;">'
            f'<span style="background:{bg};color:{fg};border:1px solid {fg};'
            f'border-radius:12px;padding:0.15rem 0.6rem;font-size:0.7rem;font-weight:700;">'
            f'✓ {detected_provider} detected</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Patient Identifier ──
    st.markdown('<span style="font-size:0.72rem;color:#aaa;">Patient Identifier</span>',
                unsafe_allow_html=True)
    col_id, col_regen = st.columns([3, 1])
    with col_id:
        patient_id = st.text_input("Patient ID", value=st.session_state["patient_id"],
                                   label_visibility="collapsed")
    with col_regen:
        if st.button("🔀", help="Generate new ID"):
            st.session_state["patient_id"] = _random_patient_id()
            st.rerun()

    st.divider()

    # ── Age  |  Biological Sex ──
    c_age, c_sex = st.columns(2)
    with c_age:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=58)
    with c_sex:
        st.markdown('<span style="font-size:0.72rem;color:#aaa;">Biological Sex</span>',
                    unsafe_allow_html=True)
        sex_label = st.radio("Sex", ["Male", "Female"], label_visibility="collapsed")
        sex = 1.0 if sex_label == "Male" else 0.0

    # ── Race  |  HbA1c ──
    c_race, c_hba1c = st.columns(2)
    with c_race:
        race = st.selectbox(
            "Race / Ethnicity Code",
            options=[0, 1, 2, 3, 4, 5],
            format_func=lambda x: str(x),
        )
    with c_hba1c:
        hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=20.0, value=8.50,
                                format="%.2f")

    # ── Diabetes  |  Foot Ulcer History ──
    c_diab, c_fu = st.columns(2)
    with c_diab:
        diab_label = st.radio("Diabetes", ["Yes", "No"])
        diabetes = 1.0 if diab_label == "Yes" else 0.0
    with c_fu:
        fu_label = st.radio("Foot Ulcer Hx", ["Yes", "No"])
        foot_ulcer_flag = 1.0 if fu_label == "Yes" else 0.0

    # ── SBP  |  DBP ──
    c_sbp, c_dbp = st.columns(2)
    with c_sbp:
        sbp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=138)
    with c_dbp:
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=85)

    # ── LDL  |  Triglycerides ──
    c_ldl, c_trig = st.columns(2)
    with c_ldl:
        ldl = st.number_input("LDL (mg/dL)", min_value=10, max_value=400, value=135)
    with c_trig:
        triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=10,
                                        max_value=1000, value=180)

    # ── HDL  |  BMI ──
    c_hdl, c_bmi = st.columns(2)
    with c_hdl:
        hdl = st.number_input("HDL (mg/dL)", min_value=5, max_value=150, value=42)
    with c_bmi:
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=28.5,
                              format="%.1f")

    # ── Creatinine  |  ACR ──
    c_cr, c_acr = st.columns(2)
    with c_cr:
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=15.0,
                                     value=1.2, format="%.1f")
    with c_acr:
        acr = st.number_input("ACR (mg/g)", min_value=0.0, max_value=1000.0, value=38.0,
                              format="%.1f")

    # ── CRP  |  eGFR ──
    c_crp, c_egfr = st.columns(2)
    with c_crp:
        crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=100.0, value=2.8,
                              format="%.1f")
    with c_egfr:
        egfr = st.number_input("eGFR (mL/min)", min_value=5.0, max_value=200.0, value=62.0,
                                format="%.1f")

    st.divider()

    # ── Image Uploads (sidebar) ──
    st.markdown('<span style="font-size:0.72rem;color:#aaa;">🖼️ Diagnostic Images</span>',
                unsafe_allow_html=True)
    retinopathy_image = st.file_uploader("Retinal Fundus Scan",
                                         type=["jpg", "jpeg", "png"],
                                         key="ret_upload")
    foot_ulcer_image = st.file_uploader("Foot Photograph",
                                        type=["jpg", "jpeg", "png"],
                                        key="foot_upload")


# ──────────────────────────────────────────────────────────────────────────────
# BUILD PATIENT VECTOR
# ──────────────────────────────────────────────────────────────────────────────
patient_vector = [
    float(age), sex, float(race), diabetes, hba1c,
    creatinine, acr, float(sbp), float(dbp), float(ldl),
    float(triglycerides), float(hdl), bmi, foot_ulcer_flag, crp, egfr,
]


# ──────────────────────────────────────────────────────────────────────────────
# MAIN AREA — Image Previews
# ──────────────────────────────────────────────────────────────────────────────
img_col1, img_col2 = st.columns(2)
with img_col1:
    if retinopathy_image:
        st.image(retinopathy_image, width=300)
        st.markdown('<p class="img-caption">📷 Uploaded Retinal Fundus Scan</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:#161b22;border:1px dashed #30363d;border-radius:10px;'
            'height:180px;display:flex;align-items:center;justify-content:center;'
            'color:#444c56;font-size:0.85rem;">👁️ Retinal Fundus Scan</div>',
            unsafe_allow_html=True,
        )

with img_col2:
    if foot_ulcer_image:
        st.image(foot_ulcer_image, width=300)
        st.markdown('<p class="img-caption">📷 Uploaded Foot Image</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:#161b22;border:1px dashed #30363d;border-radius:10px;'
            'height:180px;display:flex;align-items:center;justify-content:center;'
            'color:#444c56;font-size:0.85rem;">🦶 Foot Photograph</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# RUN BUTTON
# ──────────────────────────────────────────────────────────────────────────────
run_col, _ = st.columns([3, 1])
with run_col:
    run_clicked = st.button(
        "🚀  Run Multi-Agent Clinical Evaluation",
        type="primary",
        use_container_width=True,
    )

if run_clicked:
    with st.spinner("🔄 Multi-agent pipeline executing…"):
        output = run_agentic_mcrps(
            patient_vector,
            retinopathy_img=retinopathy_image,
            foot_ulcer_img=foot_ulcer_image,
        )
    st.session_state["pipeline_output"] = output


# ── Pipeline mode badge ──────────────────────────────────────────────────────
pipeline_output = st.session_state.get("pipeline_output")
if pipeline_output:
    mode = pipeline_output.get("mode", "deterministic")
    mode_colors = {
        "agentic":             ("#1f4a2e", "#34d399", "🤖 Agentic (CrewAI + LLM)"),
        "deterministic":       ("#1a1f2e", "#60a5fa", "⚙️ Deterministic (Offline ML)"),
        "deterministic_fallback": ("#3d2f12", "#fbbf24", "⚠️ Deterministic Fallback (LLM failed)"),
    }
    bg, fg, label = mode_colors.get(mode, ("#1a1f2e", "#94a3b8", mode))
    st.markdown(
        f'<div style="margin-bottom:0.5rem;">'
        f'  <span style="background:{bg};color:{fg};border:1px solid {fg};'
        f'        border-radius:20px;padding:0.2rem 0.8rem;font-size:0.75rem;font-weight:700;">'
        f'    {label}'
        f'  </span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if mode == "deterministic_fallback" and pipeline_output.get("error"):
        st.warning(f"LLM error: {pipeline_output['error']}")

# ── Agent Trace ──────────────────────────────────────────────────────────────
if pipeline_output:
    with st.expander("🤖 Autonomous Multi-Agent Collaboration Trace (Agents 1-6)", expanded=False):
        if pipeline_output.get("status") == "FAILED":
            st.error(f"Pipeline failed: {pipeline_output.get('error')}")
        else:
            agent_exec = pipeline_output.get("agent_execution", {})
            actions = agent_exec.get("actions_taken", [])

            trace_html = "".join(
                f'<div class="trace-line">[✓] {act}</div>' for act in actions
            )

            llm_trace = agent_exec.get("llm_trace", "")
            if llm_trace:
                trace_html += (
                    f'<div class="trace-line" style="margin-top:0.6rem;color:#8b949e;">'
                    f'── LLM Reasoning Trace ──</div>'
                    f'<div class="trace-line" style="color:#8b949e;white-space:pre-wrap;">'
                    f'{llm_trace}</div>'
                )

            st.markdown(
                f'<div class="trace-box">{trace_html}</div>',
                unsafe_allow_html=True,
            )

    # ── Main results cards ───────────────────────────────────────────────────
    if pipeline_output.get("status") != "FAILED":
        assessment = pipeline_output.get("assessment", {})
        dcrs = float(assessment.get("dcrs", 0.0))
        risk_level = assessment.get("risk_level", "UNKNOWN")
        highest_comp = assessment.get("highest_risk_complication", "Unknown")
        highest_score = float(assessment.get("highest_risk_score", 0.0))
        complication_risks = assessment.get("complication_risks", {})

        left_col, right_col = st.columns([1, 1], gap="large")

        # ── LEFT: Gauge ──────────────────────────────────────────────────────
        with left_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = _gauge_figure(dcrs)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            badge_cls = _badge_class(risk_level)
            st.markdown(
                f'<div style="text-align:center;margin-top:-0.5rem;">'
                f'  <span style="color:#8b949e;font-size:0.8rem;">Overall Risk Level &nbsp;</span>'
                f'  <span class="badge {badge_cls}">{risk_level}</span>'
                f'</div>'
                f'<div style="text-align:center;margin-top:0.5rem;font-size:0.78rem;color:#8b949e;">'
                f'  Highest: <strong style="color:#e6edf3;">{highest_comp}</strong>'
                f'  &nbsp;({highest_score:.1%})'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── RIGHT: Risk distribution bars ────────────────────────────────────
        with right_col:
            st.markdown(
                '<div class="card">'
                '<div class="card-title">📊 Complication Risk Distribution</div>',
                unsafe_allow_html=True,
            )
            if complication_risks:
                rows_html = ""
                for comp_name, risk_val in complication_risks.items():
                    val = float(risk_val) if risk_val is not None else 0.0
                    rows_html += _risk_row_html(comp_name, val)
                st.markdown(rows_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p style="color:#8b949e;font-size:0.82rem;">No risk data available.</p>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── ADA CLINICAL GUIDELINES & ACTION PLAN ──────────────────────────────
        clinical_recs = assessment.get("clinical_recommendations", {})
        if clinical_recs:
            st.markdown(
                '<div class="card">'
                '<div class="card-title">📋 ADA Clinical Guidelines & Action Plan</div>',
                unsafe_allow_html=True,
            )
            for comp, details in clinical_recs.items():
                priority = details.get("priority", "LOW")
                badge_cls = _badge_class(priority)
                st.markdown(
                    f'<div style="margin-bottom: 0.6rem;">'
                    f'<strong style="color: #e6edf3;">{comp}</strong> — '
                    f'<span class="badge {badge_cls}">Priority: {priority}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for rec in details.get("recommendations", []):
                    st.markdown(
                        f'<div style="font-size: 0.85rem; color: #c9d1d9; margin-left: 1rem; margin-bottom: 0.3rem;">• {rec}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('<hr style="margin: 0.6rem 0; border-color: #21262d;">', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Model confidence detail ───────────────────────────────────────────
        model_details = assessment.get("model_details", {})
        if model_details:
            with st.expander("🔬 Per-Model Prediction Breakdown", expanded=False):
                for comp_name, detail in model_details.items():
                    if isinstance(detail, dict) and detail.get("model_predictions"):
                        preds = detail["model_predictions"]
                        cols = st.columns(len(preds) + 1)
                        cols[0].markdown(f"**{comp_name}**")
                        for i, (mname, mval) in enumerate(preds.items()):
                            cols[i + 1].metric(
                                label=mname.upper(),
                                value=f"{float(mval):.1%}" if mval is not None else "N/A"
                            )

# ── Empty state ──────────────────────────────────────────────────────────────
if pipeline_output is None:
    st.markdown(
        '<div style="text-align:center;margin-top:3rem;color:#444c56;">'
        '<div style="font-size:3rem;">🩺</div>'
        '<div style="font-size:1rem;margin-top:0.5rem;">Fill in the patient profile and click'
        ' <strong style="color:#8b949e;">Run Multi-Agent Clinical Evaluation</strong>'
        ' to begin.</div>'
        '</div>',
        unsafe_allow_html=True,
    )