from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="RailPulse AI", page_icon="🚆", layout="wide")

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"

NAVY = "#081522"
CYAN = "#35D5E5"
AMBER = "#FFB454"
TEXT = "#EAF2FA"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{font-family:Manrope,sans-serif;}}
.stApp {{background:radial-gradient(circle at 85% 0%,#10263b 0%,{NAVY} 45%);color:{TEXT};}}
[data-testid="stSidebar"] {{background:#091725;border-right:1px solid #1d394e;}}
[data-testid="stSidebar"] .block-container {{padding:1.5rem 1rem !important;}}
.block-container {{max-width:1400px;padding-top:1.3rem;padding-bottom:2.5rem;}}
.hero {{background:linear-gradient(115deg,#102b40,#0d1b2a 65%,#103344);border:1px solid #25475d;border-radius:20px;padding:25px 28px;margin-bottom:20px;}}
.eyebrow {{color:{CYAN};font-size:.72rem;letter-spacing:.15em;font-weight:700;}}
.hero h1 {{color:#f5f9ff;font-size:2.15rem;margin:.4rem 0;font-weight:800;}}
.hero p {{color:#a9bfd1;margin:0;}}
.kpi {{background:linear-gradient(145deg,#122a3e,#0e1d2c);border:1px solid #24445b;border-radius:15px;padding:16px 18px;min-height:112px;}}
.kpi-label {{font-size:.7rem;letter-spacing:.1em;color:#9bb2c5;text-transform:uppercase;}}
.kpi-value {{font-size:1.8rem;font-weight:800;color:#f5f9ff;margin-top:5px;}}
.kpi-note {{font-size:.72rem;color:#829db3;margin-top:3px;}}
.section {{color:{CYAN};font-size:.7rem;letter-spacing:.13em;font-weight:700;margin-top:14px;}}
div[data-testid="stMetric"] {{background:#102438;border:1px solid #244158;padding:12px 14px;border-radius:13px;}}
.stTabs [data-baseweb="tab"] {{background:#102237;border-radius:9px;}}
.stTabs [aria-selected="true"] {{color:{CYAN} !important;}}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load(name):
    p = TABLES / name
    if not p.exists():
        return None
    return pd.read_csv(p)

def card(label, value, note):
    st.markdown(f"""<div class="kpi"><div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>""", unsafe_allow_html=True)

def chart_style(fig, height=340):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, family="Manrope"),
        height=height, margin=dict(l=10,r=10,t=35,b=10),
        xaxis=dict(gridcolor="#20374a"), yaxis=dict(gridcolor="#20374a"),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)")
    )
    return fig

daily = load("daily_delay_summary.csv")
bands = load("delay_categories.csv")
final_models = load("final_reserved_group_model_comparison.csv")
importance = load("final_model_permutation_importance.csv")
final_cat = load("final_reserved_group_category_analysis.csv")
pca = load("pca_explained_variance.csv")
clusters = load("delay_focused_cluster_profiles.csv")
sensitivity = load("sensitivity_experiment_results.csv")

with st.sidebar:
    st.markdown("## 🚆 RAILPULSE AI")
    st.caption("Predictive Analytics Project")
    st.markdown("---")
    page = st.radio("MENU", ["Overview", "Delay Analysis", "Model Results", "What We Learned"])
    st.markdown("---")
    st.caption("Indian Railways")
    st.caption("Dataset: September 2024")
    st.caption("Historical research dashboard")
    st.markdown("---")
    st.caption("Simple insights. Explainable predictions.")

st.markdown("""
<div class="hero">
 <div class="eyebrow">INDIAN RAILWAYS · PREDICTIVE ANALYTICS</div>
 <h1>RAILPULSE AI</h1>
 <p>Understanding train delays and predicting arrival delay at the next station.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    st.markdown('<div class="section">PROJECT AT A GLANCE</div>', unsafe_allow_html=True)
    st.subheader("A clearer picture of railway delays")
    cols = st.columns(4)
    with cols[0]: card("Model MAE", "9.49 min", "Average absolute prediction error")
    with cols[1]: card("R² Score", "0.8628", "Model fit on reserved-group evaluation")
    with cols[2]: card("Paired records", "1.10 million", "Current and next-station observations")
    with cols[3]: card("Evaluation records", "230,814", "Reserved train-group observations")

    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown('<div class="section">DAILY TREND</div>', unsafe_allow_html=True)
        st.subheader("Average delay during September 2024")
        if daily is not None:
            d = daily.copy()
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            fig = px.line(d, x="date", y="mean_delay", markers=True,
                          labels={"date":"Date","mean_delay":"Average delay (minutes)"})
            fig.update_traces(line_color=CYAN, line_width=3, marker_size=6)
            st.plotly_chart(chart_style(fig, 340), use_container_width=True)
        else:
            st.info("Daily summary is not available.")
    with right:
        st.markdown('<div class="section">DELAY CATEGORIES</div>', unsafe_allow_html=True)
        st.subheader("How delays are distributed")
        if bands is not None:
            b = bands.sort_values("percentage")
            fig = px.bar(b, x="percentage", y="delay_category", orientation="h",
                         labels={"percentage":"Share of records (%)","delay_category":"Delay range"})
            fig.update_traces(marker_color=CYAN)
            st.plotly_chart(chart_style(fig, 340), use_container_width=True)
        else:
            st.info("Delay categories are not available.")
    st.info("**In simple words:** RailPulse AI uses the delay at the current station and basic route information to estimate the arrival delay at the next station.")

elif page == "Delay Analysis":
    st.markdown('<div class="section">UNDERSTANDING THE DATA</div>', unsafe_allow_html=True)
    st.subheader("Where and how delays appear")
    if daily is not None:
        d = daily.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        if len(d):
            start, end = st.date_input("Choose dates", value=(d.date.min().date(), d.date.max().date()))
            d = d[(d.date.dt.date >= start) & (d.date.dt.date <= end)]
            a,b,c = st.columns(3)
            with a: st.metric("Days selected", len(d))
            with b: st.metric("Average daily mean", f"{d.mean_delay.mean():.2f} min" if len(d) else "—")
            with c: st.metric("Average records/day", f"{d.pair_count.mean():,.0f}" if len(d) else "—")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d.date,y=d.mean_delay,mode="lines+markers",name="Mean delay",
                                     line=dict(color=CYAN,width=3),fill="tozeroy",fillcolor="rgba(53,213,229,.10)"))
            if "median_delay" in d:
                fig.add_trace(go.Scatter(x=d.date,y=d.median_delay,mode="lines",name="Median delay",
                                         line=dict(color=AMBER,width=2,dash="dot")))
            st.plotly_chart(chart_style(fig,360),use_container_width=True)
    if bands is not None:
        st.markdown('<div class="section">DELAY BREAKDOWN</div>', unsafe_allow_html=True)
        st.subheader("Records by delay range")
        st.dataframe(bands, use_container_width=True, hide_index=True)
    st.markdown("**Key observation:** Current-station delay and next-station delay have a strong positive correlation (0.9239) in this dataset. This is an association, not proof of causation.")

elif page == "Model Results":
    st.markdown('<div class="section">PREDICTION PERFORMANCE</div>', unsafe_allow_html=True)
    st.subheader("How well does RailPulse AI perform?")
    if final_models is not None:
        st.dataframe(final_models, use_container_width=True, hide_index=True)
        if {"Model","MAE"}.issubset(final_models.columns):
            fig = px.bar(final_models, x="Model", y="MAE", color="Model",
                         color_discrete_sequence=[CYAN, AMBER],
                         labels={"MAE":"Average prediction error (minutes)"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(chart_style(fig,300),use_container_width=True)
    st.markdown("### What do these numbers mean?")
    a,b,c = st.columns(3)
    with a:
        st.metric("MAE", "9.49 minutes")
        st.caption("On average, predictions differ from actual delay by about 9.5 minutes.")
    with b:
        st.metric("RMSE", "29.10 minutes")
        st.caption("Penalizes larger prediction errors more strongly.")
    with c:
        st.metric("R²", "0.8628")
        st.caption("Indicates how much variation is explained by the model.")
    if final_cat is not None:
        st.markdown("### Prediction error across delay ranges")
        cols = [x for x in ["delay_category","persistence_MAE","delta_MAE"] if x in final_cat.columns]
        if len(cols) == 3:
            m = final_cat[cols].melt(id_vars="delay_category",var_name="model",value_name="MAE")
            m["model"] = m["model"].map({"persistence_MAE":"Simple baseline","delta_MAE":"RailPulse AI"})
            fig = px.bar(m,x="delay_category",y="MAE",color="model",barmode="group",
                         color_discrete_map={"Simple baseline":AMBER,"RailPulse AI":CYAN},
                         labels={"delay_category":"Actual delay range","MAE":"MAE (minutes)"})
            st.plotly_chart(chart_style(fig,330),use_container_width=True)
    st.warning("Evaluation note: the reserved train group was examined during earlier exploratory experiments. It is not described as a completely untouched test set.")

elif page == "What We Learned":
    st.markdown('<div class="section">MODEL EXPLANATION & TRUST</div>', unsafe_allow_html=True)
    st.subheader("What information matters most?")
    if importance is not None and {"Feature","Importance_Mean"}.issubset(importance.columns):
        friendly_names = {
            "current_arr_delay": "Current station delay",
            "distance_to_next": "Distance to next station",
            "current_distance": "Distance covered so far",
            "current_sequence": "Station sequence number",
        }
        imp = importance.copy()
        imp["Feature label"] = imp["Feature"].map(friendly_names).fillna(imp["Feature"])
        imp = imp.sort_values("Importance_Mean")
        fig = px.bar(imp,x="Importance_Mean",y="Feature label",orientation="h",
                     labels={"Importance_Mean":"Increase in prediction error after shuffling","Feature label":"Input information"})
        fig.update_traces(marker_color=CYAN)
        st.plotly_chart(chart_style(fig,340),use_container_width=True)
        st.caption("This chart explains the final four-feature model. A larger value means shuffling that input increased validation error more.")
        st.info("Current station delay is the most influential input in this experiment. The other three route-related inputs also contribute.")
    st.markdown("### Exploring patterns")
    a,b = st.columns(2)
    with a:
        if pca is not None:
            st.markdown("**PCA: simplifying information**")
            total = pca["Explained_Variance_Percent"].sum()
            st.metric("Variance captured by first two components",f"{total:.2f}%")
            st.caption("PCA summarizes patterns in the selected features using two new components.")
    with b:
        if clusters is not None:
            st.markdown("**K-means: grouping similar observations**")
            st.metric("Exploratory groups", "6")
            st.caption("The six groups describe patterns of current delay and distance to the next station. They are not official railway categories.")
    st.markdown("### Data quality check")
    if sensitivity is not None and len(sensitivity):
        s = sensitivity.iloc[0]
        a,b,c = st.columns(3)
        with a: st.metric("Affected rows excluded", f"{int(s['excluded_affected_rows'])}")
        with b: st.metric("Clean-data MAE", f"{s['MAE']:.4f} min")
        with c: st.metric("Clean-data R²", f"{s['R2']:.4f}")
        st.success("Removing the 183 identified affected rows changed model performance only slightly.")
    st.caption("Timestamp records with uncertain calendar-day offsets were not automatically corrected. The clean-data run is a separate sensitivity experiment.")

st.markdown("---")
st.caption("RAILPULSE AI · Student predictive analytics research prototype · Historical September 2024 data · Not a live train tracking service")
