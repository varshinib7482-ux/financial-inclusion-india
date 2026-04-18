# ============================================================
# FINANCIAL INCLUSION INDIA — STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Inclusion India",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #16213e, #0f3460);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stSelectbox label {
        font-weight: 600;
    }
    .sidebar .sidebar-content {
        background: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    master    = pd.read_csv('FI_India_Master_Dataset.csv')
    gap       = pd.read_csv('gap_analysis.csv')
    topsis    = pd.read_csv('topsis_ranking.csv')
    clusters  = pd.read_csv('state_clusters.csv')
    return master, gap, topsis, clusters

# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(master):
    features = [
        'hdi', 'health_index', 'income_index',
        'education_index', 'wealth_index', 'poverty_rate',
        'electricity_access', 'piped_water',
        'avg_years_education', 'primary_attendance',
        'secondary_attendance'
    ]

    X = master[features].copy()
    y_reg = master['fii_score'].copy()

    le = LabelEncoder()
    y_clf = le.fit_transform(master['fii_tier'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )

    rf_reg = RandomForestRegressor(
        n_estimators=200, random_state=42,
        max_depth=6, min_samples_leaf=5
    )
    rf_reg.fit(X_train, y_train)

    rf_clf = RandomForestClassifier(
        n_estimators=200, random_state=42,
        max_depth=6, min_samples_leaf=5
    )
    rf_clf.fit(X_train_c, y_train_c)

    return rf_reg, rf_clf, le, features

master, gap, topsis, clusters = load_data()
rf_reg, rf_clf, le, features = train_models(master)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("""
<div style='text-align:center; padding:1rem;
background:linear-gradient(135deg,#1a1a2e,#0f3460);
border-radius:10px; margin-bottom:1rem;'>
<h2 style='color:white; margin:0;'>🏦 FI India</h2>
<p style='color:#aaa; margin:0; font-size:0.8rem;'>
Financial Inclusion Dashboard</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview Dashboard",
        "🔍 State Deep Dive",
        "🤖 Prediction Engine",
        "🗂️ Cluster Explorer",
        "📊 Gap & TOPSIS Analysis",
        "💡 Policy Recommendations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.75rem; color:#888; padding:0.5rem;'>
<b>Data Sources:</b><br>
RBI Handbook of Statistics<br>
Global Data Lab SHDI v10.1<br>
<br>
<b>Coverage:</b><br>
32 Indian States<br>
2004 — 2014<br>
363 Observations
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE 1 — OVERVIEW DASHBOARD
# ============================================================

if page == "🏠 Overview Dashboard":

    st.markdown("""
    <div class='main-header'>
    🏦 Financial Inclusion Across Indian States
    <br><small style='font-size:1rem; font-weight:400;'>
    A Data-Driven Policy Analysis (2004–2014)</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    state_avg = master.groupby('State')['fii_score'].mean()

    with col1:
        st.metric(
            "States Analysed", "32",
            help="Indian States and UTs"
        )
    with col2:
        st.metric(
            "Years Covered", "2004–2014",
            help="11 years of panel data"
        )
    with col3:
        st.metric(
            "Highest FII",
            f"{state_avg.idxmax()}",
            f"{state_avg.max():.3f}"
        )
    with col4:
        st.metric(
            "Lowest FII",
            f"{state_avg.idxmin()}",
            f"{state_avg.min():.3f}"
        )

    st.markdown("---")

    # FII by state bar chart
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📊 Financial Inclusion Index by State")

        state_fii = master.groupby('State')['fii_score']\
            .mean().sort_values(ascending=True).reset_index()

        tier_map = master.groupby('State')['fii_tier']\
            .agg(lambda x: x.mode()[0])

        color_map = {
            'High'  : '#2ecc71',
            'Medium': '#f39c12',
            'Low'   : '#e74c3c'
        }

        state_fii['tier'] = state_fii['State'].map(tier_map)
        state_fii['color'] = state_fii['tier'].map(color_map)

        fig = px.bar(
            state_fii,
            x='fii_score',
            y='State',
            orientation='h',
            color='tier',
            color_discrete_map=color_map,
            title='Average FII Score by State (2004–2014)',
            labels={
                'fii_score': 'FII Score',
                'State': '',
                'tier': 'Inclusion Tier'
            },
            height=700
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=11)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🏆 Top 5 States")
        top5 = state_avg.sort_values(
            ascending=False
        ).head(5)
        for i, (state, score) in enumerate(top5.items()):
            st.markdown(f"""
            <div class='insight-box'>
            <b>#{i+1} {state}</b><br>
            FII Score: {score:.4f}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("###")
        st.subheader("⚠️ Bottom 5 States")
        bottom5 = state_avg.sort_values(
            ascending=True
        ).head(5)
        for i, (state, score) in enumerate(bottom5.items()):
            st.markdown(f"""
            <div class='insight-box'>
            <b>#{i+1} {state}</b><br>
            FII Score: {score:.4f}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # FII trend over time
    st.subheader("📈 Financial Inclusion Trend (2004–2014)")

    top5_list = state_avg.sort_values(
        ascending=False
    ).head(5).index.tolist()
    bottom5_list = state_avg.sort_values(
        ascending=True
    ).head(5).index.tolist()

    selected_states = top5_list + bottom5_list
    trend_data = master[
        master['State'].isin(selected_states)
    ].groupby(['State', 'Year'])['fii_score']\
     .mean().reset_index()

    fig2 = px.line(
        trend_data,
        x='Year',
        y='fii_score',
        color='State',
        title='FII Score Trend — Top 5 vs Bottom 5 States',
        labels={
            'fii_score': 'FII Score',
            'Year': 'Year'
        },
        height=450,
        markers=True
    )
    fig2.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PAGE 2 — STATE DEEP DIVE
# ============================================================

elif page == "🔍 State Deep Dive":

    st.markdown("""
    <div class='main-header'>
    🔍 State Deep Dive Analysis
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###")

    selected_state = st.selectbox(
        "Select a State",
        sorted(master['State'].unique())
    )

    state_data = master[
        master['State'] == selected_state
    ].mean(numeric_only=True)

    state_tier = master[
        master['State'] == selected_state
    ]['fii_tier'].mode()[0]

    tier_color = {
        'High': '🟢', 'Medium': '🟡', 'Low': '🔴'
    }

    st.markdown(f"## {selected_state}")
    st.markdown(
        f"**Inclusion Tier:** "
        f"{tier_color[state_tier]} {state_tier}"
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("FII Score",
                  f"{state_data['fii_score']:.4f}")
    with col2:
        st.metric("HDI Score",
                  f"{state_data['hdi']:.3f}")
    with col3:
        st.metric("Poverty Rate",
                  f"{state_data['poverty_rate']:.1f}%")
    with col4:
        st.metric("Electricity Access",
                  f"{state_data['electricity_access']:.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        # Radar chart
        st.subheader("📡 State Profile — Radar Chart")

        scaler_radar = MinMaxScaler()
        radar_cols = [
            'hdi', 'wealth_index', 'electricity_access',
            'avg_years_education', 'primary_attendance',
            'fii_score'
        ]
        radar_labels = [
            'HDI', 'Wealth', 'Electricity',
            'Education Yrs', 'School Attendance',
            'FII Score'
        ]

        all_states_avg = master.groupby('State')[
            radar_cols
        ].mean()

        scaled = pd.DataFrame(
            scaler_radar.fit_transform(all_states_avg),
            columns=radar_cols,
            index=all_states_avg.index
        )

        state_radar = scaled.loc[selected_state].values.tolist()
        state_radar += state_radar[:1]

        angles = np.linspace(
            0, 2*np.pi,
            len(radar_labels),
            endpoint=False
        ).tolist()
        angles += angles[:1]

        fig_radar, ax = plt.subplots(
            figsize=(6, 6),
            subplot_kw=dict(polar=True)
        )

        ax.plot(angles, state_radar,
                'o-', linewidth=2, color='#0f3460')
        ax.fill(angles, state_radar,
                alpha=0.25, color='#0f3460')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, size=10)
        ax.set_ylim(0, 1)
        ax.set_title(
            f'{selected_state} — Development Profile',
            size=12, fontweight='bold', pad=20
        )
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_radar)

    with col_right:
        # FII trend for selected state
        st.subheader("📈 FII Trend Over Time")

        state_trend = master[
            master['State'] == selected_state
        ].groupby('Year')['fii_score'].mean().reset_index()

        fig_trend = px.line(
            state_trend,
            x='Year',
            y='fii_score',
            title=f'{selected_state} — FII Score 2004–2014',
            labels={'fii_score': 'FII Score'},
            markers=True,
            height=350
        )
        fig_trend.update_traces(
            line_color='#0f3460',
            marker_color='#e74c3c'
        )
        fig_trend.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Compare with national average
        st.subheader("🔄 vs National Average")

        nat_avg = master.groupby('Year')[
            'fii_score'
        ].mean().reset_index()
        nat_avg['Type'] = 'National Average'

        state_trend['Type'] = selected_state

        combined = pd.concat([state_trend, nat_avg])

        fig_comp = px.line(
            combined,
            x='Year',
            y='fii_score',
            color='Type',
            title='State vs National Average',
            markers=True,
            height=300
        )
        fig_comp.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_comp, use_container_width=True)

# ============================================================
# PAGE 3 — PREDICTION ENGINE
# ============================================================

elif page == "🤖 Prediction Engine":

    st.markdown("""
    <div class='main-header'>
    🤖 Financial Inclusion Prediction Engine
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###")
    st.markdown("""
    <div class='insight-box'>
    Enter socioeconomic indicators for any state or 
    hypothetical scenario to predict its Financial 
    Inclusion Index score and tier classification.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("###")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🧬 Development Indicators")
        hdi = st.slider(
            "HDI Score", 0.3, 0.95, 0.65, 0.01
        )
        health_index = st.slider(
            "Health Index", 0.3, 0.95, 0.70, 0.01
        )
        income_index = st.slider(
            "Income Index", 0.3, 0.95, 0.60, 0.01
        )
        education_index = st.slider(
            "Education Index", 0.2, 0.95, 0.55, 0.01
        )

    with col2:
        st.subheader("💰 Wealth & Poverty")
        wealth_index = st.slider(
            "Wealth Index", 10.0, 80.0, 45.0, 0.5
        )
        poverty_rate = st.slider(
            "Poverty Rate (%)", 5.0, 90.0, 45.0, 0.5
        )
        electricity_access = st.slider(
            "Electricity Access (%)", 20.0, 100.0, 80.0, 0.5
        )
        piped_water = st.slider(
            "Piped Water Access (%)", 1.0, 90.0, 40.0, 0.5
        )

    with col3:
        st.subheader("📚 Education")
        avg_years_education = st.slider(
            "Avg Years of Education", 1.0, 12.0, 5.0, 0.1
        )
        primary_attendance = st.slider(
            "Primary School Attendance (%)",
            30.0, 100.0, 80.0, 0.5
        )
        secondary_attendance = st.slider(
            "Secondary School Attendance (%)",
            30.0, 100.0, 75.0, 0.5
        )

    st.markdown("---")

    if st.button("🔮 Predict Financial Inclusion",
                 use_container_width=True):

        input_data = np.array([[
            hdi, health_index, income_index,
            education_index, wealth_index, poverty_rate,
            electricity_access, piped_water,
            avg_years_education, primary_attendance,
            secondary_attendance
        ]])

        fii_pred   = rf_reg.predict(input_data)[0]
        tier_pred  = le.inverse_transform(
            rf_clf.predict(input_data)
        )[0]
        tier_proba = rf_clf.predict_proba(input_data)[0]

        fii_pred = np.clip(fii_pred, 0, 1)

        col_r1, col_r2, col_r3 = st.columns(3)

        tier_emoji = {
            'High': '🟢', 'Medium': '🟡', 'Low': '🔴'
        }

        with col_r1:
            st.metric(
                "Predicted FII Score",
                f"{fii_pred:.4f}",
                help="0 = No inclusion, 1 = Full inclusion"
            )
        with col_r2:
            st.metric(
                "Inclusion Tier",
                f"{tier_emoji.get(tier_pred, '')} "
                f"{tier_pred}"
            )
        with col_r3:
            nat_mean = master['fii_score'].mean()
            diff = fii_pred - nat_mean
            st.metric(
                "vs National Average",
                f"{fii_pred:.4f}",
                f"{diff:+.4f}"
            )

        st.markdown("###")

        # Probability breakdown
        st.subheader("📊 Tier Probability Breakdown")

        proba_df = pd.DataFrame({
            'Tier'       : le.classes_,
            'Probability': tier_proba * 100
        }).sort_values('Probability', ascending=False)

        fig_proba = px.bar(
            proba_df,
            x='Tier',
            y='Probability',
            color='Tier',
            color_discrete_map={
                'High'  : '#2ecc71',
                'Low'   : '#e74c3c',
                'Medium': '#f39c12'
            },
            title='Probability of Each Inclusion Tier',
            labels={'Probability': 'Probability (%)'},
            height=350
        )
        fig_proba.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_proba, use_container_width=True)

        # Key drivers
        st.subheader("🔑 Key Drivers of This Prediction")

        importances = rf_reg.feature_importances_
        driver_df = pd.DataFrame({
            'Factor'    : features,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(5)

        for _, row in driver_df.iterrows():
            st.progress(
                float(row['Importance']),
                text=f"{row['Factor']} — "
                     f"{row['Importance']*100:.1f}%"
            )

# ============================================================
# PAGE 4 — CLUSTER EXPLORER
# ============================================================

elif page == "🗂️ Cluster Explorer":

    st.markdown("""
    <div class='main-header'>
    🗂️ State Cluster Explorer
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###")

    cluster_colors = {
        'Financially Advanced'  : '#2ecc71',
        'Developing Inclusion'  : '#3498db',
        'Emerging Inclusion'    : '#f39c12',
        'Financially Excluded'  : '#e74c3c'
    }

    # Cluster summary metrics
    col1, col2, col3, col4 = st.columns(4)

    for i, (cluster, color) in enumerate(
        cluster_colors.items()
    ):
        count = len(
            clusters[clusters['Cluster_Name'] == cluster]
        )
        avg_fii = clusters[
            clusters['Cluster_Name'] == cluster
        ]['fii_score'].mean()

        cols = [col1, col2, col3, col4]
        with cols[i]:
            st.markdown(f"""
            <div style='background:{color}22;
            border-left:4px solid {color};
            padding:0.8rem; border-radius:5px;'>
            <b style='color:{color};'>{cluster}</b><br>
            {count} States<br>
            Avg FII: {avg_fii:.3f}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # PCA scatter plot
    st.subheader("🔵 Cluster Visualization (PCA)")

    fig_cluster = px.scatter(
        clusters,
        x='PCA1',
        y='PCA2',
        color='Cluster_Name',
        color_discrete_map=cluster_colors,
        text='State',
        title='K-Means Cluster Analysis — Indian States',
        labels={
            'PCA1': 'Principal Component 1',
            'PCA2': 'Principal Component 2',
            'Cluster_Name': 'Cluster'
        },
        height=600,
        size_max=15
    )
    fig_cluster.update_traces(
        textposition='top center',
        marker=dict(size=12, opacity=0.8)
    )
    fig_cluster.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("---")

    # States by cluster
    st.subheader("📋 States by Cluster")

    for cluster, color in cluster_colors.items():
        with st.expander(f"{cluster}"):
            states_in_cluster = clusters[
                clusters['Cluster_Name'] == cluster
            ][['State', 'fii_score', 'hdi', 'poverty_rate']]

            st.dataframe(
                states_in_cluster.sort_values(
                    'fii_score', ascending=False
                ).reset_index(drop=True),
                use_container_width=True
            )

# ============================================================
# PAGE 5 — GAP & TOPSIS ANALYSIS
# ============================================================

elif page == "📊 Gap & TOPSIS Analysis":

    st.markdown("""
    <div class='main-header'>
    📊 Gap Analysis & TOPSIS Ranking
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###")

    tab1, tab2 = st.tabs([
        "📉 Gap Analysis",
        "🏆 TOPSIS Ranking"
    ])

    with tab1:
        st.subheader("Financial Inclusion Gap Analysis")
        st.markdown("""
        <div class='insight-box'>
        <b>Gap = Actual FII − Predicted FII</b><br>
        🔴 Negative gap = Underperforming 
        (banking access below development potential)<br>
        🟢 Positive gap = Overperforming 
        (banking access exceeds development level)<br>
        🔵 Near zero = On Track
        </div>
        """, unsafe_allow_html=True)

        st.markdown("###")

        color_map_gap = {
            'Underperforming': '#e74c3c',
            'On Track'       : '#3498db',
            'Overperforming' : '#2ecc71'
        }

        gap_sorted = gap.sort_values('fii_gap')

        fig_gap = px.bar(
            gap_sorted,
            x='fii_gap',
            y='State',
            orientation='h',
            color='performance',
            color_discrete_map=color_map_gap,
            title='FII Gap by State',
            labels={
                'fii_gap'    : 'FII Gap',
                'State'      : '',
                'performance': 'Performance'
            },
            height=750
        )
        fig_gap.add_vline(
            x=0, line_dash="dash",
            line_color="black", line_width=2
        )
        fig_gap.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_gap, use_container_width=True)

        st.dataframe(
            gap[[
                'State', 'actual_fii',
                'predicted_fii', 'fii_gap',
                'performance', 'hdi', 'poverty_rate'
            ]].sort_values(
                'fii_gap'
            ).reset_index(drop=True),
            use_container_width=True
        )

    with tab2:
        st.subheader("TOPSIS — State Priority Ranking")
        st.markdown("""
        <div class='insight-box'>
        <b>TOPSIS (Multi Criteria Decision Making)</b> ranks 
        states by urgency of financial inclusion intervention.
        Higher priority states need immediate policy attention.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("###")

        priority_colors = {
            'High Priority'  : '#e74c3c',
            'Medium Priority': '#f39c12',
            'Low Priority'   : '#2ecc71'
        }

        fig_topsis = px.bar(
            topsis.sort_values(
                'topsis_score', ascending=True
            ),
            x='topsis_score',
            y='State',
            orientation='h',
            color='priority_tier',
            color_discrete_map=priority_colors,
            title='TOPSIS Priority Ranking — '
                  'States Needing Intervention',
            labels={
                'topsis_score' : 'TOPSIS Score',
                'State'        : '',
                'priority_tier': 'Priority'
            },
            height=750
        )
        fig_topsis.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_topsis, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🔴 High Priority")
            high = topsis[
                topsis['priority_tier'] == 'High Priority'
            ][['State', 'topsis_rank', 'poverty_rate']]
            st.dataframe(high, use_container_width=True)

        with col2:
            st.markdown("### 🟡 Medium Priority")
            med = topsis[
                topsis['priority_tier'] == 'Medium Priority'
            ][['State', 'topsis_rank', 'poverty_rate']]
            st.dataframe(med, use_container_width=True)

        with col3:
            st.markdown("### 🟢 Low Priority")
            low = topsis[
                topsis['priority_tier'] == 'Low Priority'
            ][['State', 'topsis_rank', 'poverty_rate']]
            st.dataframe(low, use_container_width=True)

# ============================================================
# PAGE 6 — POLICY RECOMMENDATIONS
# ============================================================

elif page == "💡 Policy Recommendations":

    st.markdown("""
    <div class='main-header'>
    💡 Policy Recommendations
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###")

    st.markdown("""
    <div class='insight-box'>
    <h3>🎯 Key Finding</h3>
    Primary school attendance is the #1 predictor of 
    financial inclusion (29.9% importance). 
    States that invest in education today will have 
    better banking access tomorrow.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🏦 For RBI & Banking Sector

        **1. Priority Branch Expansion**
        Focus new branch approvals in Bihar, Jharkhand,
        Assam and northeastern states — identified as
        High Priority by TOPSIS analysis.

        **2. Business Correspondent Model**
        Strengthen BC network in states with high poverty
        rates (>75%) — UP, West Bengal, Odisha, MP.

        **3. Digital Banking Push**
        States like Punjab and Haryana are underperforming
        relative to their development level — mobile
        banking and UPI adoption campaigns needed.

        **4. Targeted Credit Schemes**
        Rajasthan and UP show high credit-deposit ratios
        despite poverty — scale successful models to
        similar states.
        """)

    with col2:
        st.markdown("""
        ### 🏛️ For State Governments & NITI Aayog

        **1. Education Investment**
        Primary school attendance drives financial inclusion
        more than any other variable. Every 1% increase in
        school attendance improves inclusion potential.

        **2. Infrastructure Development**
        Piped water and electricity access are strong
        predictors — basic infrastructure precedes
        financial inclusion.

        **3. Northeast Special Package**
        Nagaland, Arunachal Pradesh, Manipur and Meghalaya
        need dedicated financial inclusion packages
        combining infrastructure + banking + education.

        **4. Kerala Model Study**
        Kerala has high HDI but medium FII — study
        barriers to formal banking despite high literacy
        and replicate solutions elsewhere.
        """)

    st.markdown("---")

    st.subheader("📊 Model Performance Summary")

    perf_data = {
        'Model'    : [
            'Linear Regression',
            'Ridge Regression',
            'Random Forest Regressor',
            'Logistic Regression',
            'Random Forest Classifier'
        ],
        'Type'     : [
            'Regression', 'Regression', 'Regression',
            'Classification', 'Classification'
        ],
        'R² / AUC' : [
            '0.4366', '0.3750', '0.9047',
            '0.8223 AUC', '0.9545 AUC'
        ],
        'Verdict'  : [
            'Baseline', 'Below Baseline', 'Best Model ⭐',
            'Good Baseline', 'Best Model ⭐'
        ]
    }

    st.dataframe(
        pd.DataFrame(perf_data),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("⚠️ Limitations")
    st.markdown("""
    - Dataset covers 2004–2014 (pre-UPI, pre-Jan Dhan maturity)
    - State name standardization may have minor discrepancies
    - Wealth and Education data averaged from 2006 and 2012 surveys
    - FII is a composite index — weightings are equal across components
    - Results reflect structural patterns, not real-time banking status
    """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#888; 
    font-size:0.85rem;'>
    Built for MBA Applied Business Analytics Project<br>
    Data: RBI Handbook of Statistics + Global Data Lab SHDI v10.1
    </div>
    """, unsafe_allow_html=True)