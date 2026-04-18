# ============================================================
# FINANCIAL INCLUSION INDIA — STREAMLIT APP
# ============================================================


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Financial Inclusion India",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.big-header {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #1a237e, #0d47a1, #01579b);
    color: white;
    border-radius: 15px;
    margin-bottom: 2rem;
    border: 1px solid #1565c0;
}
.big-header h1 { font-size: 2.4rem; font-weight: 900; margin: 0; color: white; }
.big-header p { font-size: 1rem; margin: 0.5rem 0 0 0; color: #90caf9; }
.kpi-card {
    background: linear-gradient(135deg, #1a237e, #283593);
    border: 1px solid #3949ab;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: white;
}
.kpi-card .kpi-value {
    font-size: 1.8rem;
    font-weight: 900;
    color: #64b5f6;
}
.kpi-card .kpi-label {
    font-size: 0.85rem;
    color: #90caf9;
    margin-top: 0.3rem;
}
.state-card-green {
    background: #0a2e0a;
    border: 1.5px solid #2ecc71;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    color: white;
}
.state-card-red {
    background: #2e0a0a;
    border: 1.5px solid #e74c3c;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    color: white;
}
.insight-box {
    background: #1a237e22;
    border-left: 4px solid #42a5f5;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    margin: 0.8rem 0;
    color: white;
    font-size: 0.95rem;
}
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #64b5f6;
    margin: 1rem 0 0.5rem 0;
    border-bottom: 2px solid #1565c0;
    padding-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    master   = pd.read_csv('FI_India_Master_Dataset.csv')
    gap      = pd.read_csv('gap_analysis.csv')
    topsis   = pd.read_csv('topsis_ranking.csv')
    clusters = pd.read_csv('state_clusters.csv')
    return master, gap, topsis, clusters

@st.cache_resource
def train_models(master):
    features = [
        'hdi', 'health_index', 'income_index',
        'education_index', 'wealth_index', 'poverty_rate',
        'electricity_access', 'piped_water',
        'avg_years_education', 'primary_attendance',
        'secondary_attendance'
    ]
    X     = master[features].copy()
    y_reg = master['fii_score'].copy()
    le    = LabelEncoder()
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
rf_reg, rf_clf, le, features  = train_models(master)

st.sidebar.markdown("""
<div style='text-align:center; padding:1.2rem;
background:linear-gradient(135deg,#1a237e,#0d47a1);
border-radius:12px; margin-bottom:1rem;
border:1px solid #1565c0;'>
<div style='font-size:2rem;'>🏦</div>
<div style='color:white; font-size:1.1rem; font-weight:700;'>FI India</div>
<div style='color:#90caf9; font-size:0.78rem;'>Financial Inclusion Dashboard</div>
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
<div style='font-size:0.8rem; color:#90caf9; padding:0.5rem;
background:#0d1b3e; border-radius:8px; border:1px solid #1a3a6e;'>
<b style='color:#64b5f6;'>Data Sources</b><br>
RBI Handbook of Statistics<br>
Global Data Lab SHDI v10.1<br><br>
<b style='color:#64b5f6;'>Coverage</b><br>
32 Indian States<br>
2004 — 2014<br>
363 Observations<br><br>
<b style='color:#64b5f6;'>Models</b><br>
Random Forest R² = 0.90<br>
AUC-ROC = 0.955
</div>
""", unsafe_allow_html=True)

if page == "🏠 Overview Dashboard":

    st.markdown("""
    <div class='big-header'>
    <h1>🏦 Financial Inclusion Across Indian States</h1>
    <p>A Data-Driven Policy Analysis (2004–2014) | MBA Applied Business Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    state_avg = master.groupby('State')['fii_score'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='kpi-card'>
        <div class='kpi-value'>32</div>
        <div class='kpi-label'>States Analysed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='kpi-card'>
        <div class='kpi-value'>2004–2014</div>
        <div class='kpi-label'>Years Covered</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='kpi-card'>
        <div class='kpi-value'>{state_avg.idxmax()}</div>
        <div class='kpi-label'>Highest FII — {state_avg.max():.3f}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='kpi-card'>
        <div class='kpi-value'>{state_avg.idxmin()}</div>
        <div class='kpi-label'>Lowest FII — {state_avg.min():.3f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📊 Financial Inclusion Index by State</div>",
                unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        state_fii = master.groupby('State')['fii_score']\
            .mean().sort_values(ascending=True).reset_index()
        tier_map  = master.groupby('State')['fii_tier']\
            .agg(lambda x: x.mode()[0])
        color_map = {
            'High'  : '#2ecc71',
            'Medium': '#f39c12',
            'Low'   : '#e74c3c'
        }
        state_fii['tier']  = state_fii['State'].map(tier_map)
        state_fii['color'] = state_fii['tier'].map(color_map)

        fig = px.bar(
            state_fii,
            x='fii_score', y='State',
            orientation='h',
            color='tier',
            color_discrete_map=color_map,
            labels={
                'fii_score': 'FII Score',
                'State'    : '',
                'tier'     : 'Inclusion Tier'
            },
            height=700
        )
        fig.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white', size=12),
            xaxis=dict(
                gridcolor='#1f2937',
                color='white',
                title_font=dict(color='white', size=13)
            ),
            yaxis=dict(
                color='white',
                tickfont=dict(size=11, color='white')
            ),
            legend=dict(
                font=dict(color='white'),
                bgcolor='#1a237e',
                bordercolor='#3949ab'
            ),
            title=dict(
                text='Average FII Score by State (2004–2014)',
                font=dict(color='white', size=15),
                x=0.5
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>🏆 Top 5 States</div>",
                    unsafe_allow_html=True)
        medals = ['🥇','🥈','🥉','4️⃣','5️⃣']
        top5   = state_avg.sort_values(ascending=False).head(5)
        for i, (state, score) in enumerate(top5.items()):
            st.markdown(f"""
            <div class='state-card-green'>
            <span style='font-size:1.3rem;'>{medals[i]}</span>
            <b style='color:white; font-size:1rem;'> {state}</b><br>
            <span style='color:#2ecc71; font-size:0.9rem;
            font-weight:600;'>FII Score: {score:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>⚠️ Bottom 5 States</div>",
                    unsafe_allow_html=True)
        bottom5 = state_avg.sort_values(ascending=True).head(5)
        for i, (state, score) in enumerate(bottom5.items()):
            st.markdown(f"""
            <div class='state-card-red'>
            <span style='font-size:1.1rem;'>🔴</span>
            <b style='color:white; font-size:1rem;'> {state}</b><br>
            <span style='color:#e74c3c; font-size:0.9rem;
            font-weight:600;'>FII Score: {score:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📈 Financial Inclusion Trend (2004–2014)</div>",
                unsafe_allow_html=True)

    top5_list    = state_avg.sort_values(ascending=False).head(5).index.tolist()
    bottom5_list = state_avg.sort_values(ascending=True).head(5).index.tolist()
    trend_data   = master[
        master['State'].isin(top5_list + bottom5_list)
    ].groupby(['State','Year'])['fii_score'].mean().reset_index()

    fig2 = px.line(
        trend_data, x='Year', y='fii_score',
        color='State',
        markers=True,
        labels={'fii_score':'FII Score', 'Year':'Year'},
        height=420
    )
    fig2.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='#1f2937', color='white',
                   tickfont=dict(color='white')),
        yaxis=dict(gridcolor='#1f2937', color='white',
                   tickfont=dict(color='white')),
        legend=dict(font=dict(color='white'),
                    bgcolor='#1a237e',
                    bordercolor='#3949ab'),
        title=dict(
            text='FII Score Trend — Top 5 vs Bottom 5 States',
            font=dict(color='white', size=15), x=0.5
        )
    )
    st.plotly_chart(fig2, use_container_width=True)

elif page == "🔍 State Deep Dive":

    st.markdown("""
    <div class='big-header'>
    <h1>🔍 State Deep Dive Analysis</h1>
    <p>Explore detailed financial inclusion profile for any Indian state</p>
    </div>
    """, unsafe_allow_html=True)

    selected_state = st.selectbox(
        "Select a State",
        sorted(master['State'].unique())
    )

    state_data = master[master['State']==selected_state]\
        .mean(numeric_only=True)
    state_tier = master[master['State']==selected_state]\
        ['fii_tier'].mode()[0]

    tier_color = {'High':'#2ecc71','Medium':'#f39c12','Low':'#e74c3c'}
    tier_emoji = {'High':'🟢','Medium':'🟡','Low':'🔴'}

    st.markdown(f"""
    <div style='background:#1a237e; border:1px solid #3949ab;
    border-radius:12px; padding:1rem 1.5rem; margin:1rem 0;
    display:flex; align-items:center; gap:1rem;'>
    <div style='font-size:2rem;'>🗺️</div>
    <div>
    <div style='font-size:1.5rem; font-weight:800;
    color:white;'>{selected_state}</div>
    <div style='color:{tier_color[state_tier]};
    font-weight:600; font-size:1rem;'>
    {tier_emoji[state_tier]} {state_tier} Inclusion Tier</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("FII Score", f"{state_data['fii_score']:.4f}"),
        ("HDI Score", f"{state_data['hdi']:.3f}"),
        ("Poverty Rate", f"{state_data['poverty_rate']:.1f}%"),
        ("Electricity", f"{state_data['electricity_access']:.1f}%")
    ]
    for col, (label, value) in zip([col1,col2,col3,col4], metrics):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div class='section-header'>📡 State Profile Radar</div>",
                    unsafe_allow_html=True)
        scaler_r  = MinMaxScaler()
        radar_cols = [
            'hdi','wealth_index','electricity_access',
            'avg_years_education','primary_attendance','fii_score'
        ]
        radar_labels = [
            'HDI','Wealth','Electricity',
            'Education','School Att.','FII Score'
        ]
        all_avg = master.groupby('State')[radar_cols].mean()
        scaled  = pd.DataFrame(
            scaler_r.fit_transform(all_avg),
            columns=radar_cols, index=all_avg.index
        )
        vals   = scaled.loc[selected_state].values.tolist()
        vals  += vals[:1]
        angles = np.linspace(0,2*np.pi,len(radar_labels),
                             endpoint=False).tolist()
        angles += angles[:1]

        fig_r, ax = plt.subplots(
            figsize=(5,5), subplot_kw=dict(polar=True)
        )
        fig_r.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.plot(angles, vals, 'o-', linewidth=2.5,
                color='#64b5f6')
        ax.fill(angles, vals, alpha=0.3, color='#1565c0')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, size=11,
                           color='white', fontweight='bold')
        ax.set_ylim(0,1)
        ax.tick_params(colors='white')
        ax.spines['polar'].set_color('#1565c0')
        ax.yaxis.set_tick_params(labelcolor='#90caf9')
        ax.grid(color='#1f2937', linewidth=0.8)
        ax.set_title(
            f'{selected_state}',
            size=13, fontweight='bold', pad=20, color='white'
        )
        st.pyplot(fig_r)

    with col_right:
        st.markdown("<div class='section-header'>📈 FII Trend</div>",
                    unsafe_allow_html=True)
        state_trend = master[master['State']==selected_state]\
            .groupby('Year')['fii_score'].mean().reset_index()

        fig_t = px.line(
            state_trend, x='Year', y='fii_score',
            markers=True, height=280,
            labels={'fii_score':'FII Score'}
        )
        fig_t.update_traces(
            line_color='#64b5f6',
            marker_color='#f39c12',
            marker_size=8
        )
        fig_t.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#1f2937', color='white',
                       tickfont=dict(color='white')),
            yaxis=dict(gridcolor='#1f2937', color='white',
                       tickfont=dict(color='white')),
            title=dict(
                text=f'{selected_state} FII 2004–2014',
                font=dict(color='white',size=13), x=0.5
            )
        )
        st.plotly_chart(fig_t, use_container_width=True)

        st.markdown("<div class='section-header'>🔄 vs National Average</div>",
                    unsafe_allow_html=True)
        nat  = master.groupby('Year')['fii_score']\
            .mean().reset_index()
        nat['Type']          = 'National Average'
        state_trend['Type']  = selected_state
        combined             = pd.concat([state_trend, nat])

        fig_c = px.line(
            combined, x='Year', y='fii_score',
            color='Type', markers=True, height=270,
            color_discrete_map={
                selected_state : '#64b5f6',
                'National Average': '#f39c12'
            },
            labels={'fii_score':'FII Score'}
        )
        fig_c.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#1f2937', color='white',
                       tickfont=dict(color='white')),
            yaxis=dict(gridcolor='#1f2937', color='white',
                       tickfont=dict(color='white')),
            legend=dict(font=dict(color='white'),
                        bgcolor='#1a237e')
        )
        st.plotly_chart(fig_c, use_container_width=True)

elif page == "🤖 Prediction Engine":

    st.markdown("""
    <div class='big-header'>
    <h1>🤖 Financial Inclusion Prediction Engine</h1>
    <p>Input socioeconomic indicators to predict a state's FII Score and Inclusion Tier</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
    Adjust the sliders below to simulate any state's socioeconomic profile.
    The Random Forest model (R² = 0.90) will predict its Financial Inclusion
    Index score and classify it into High / Medium / Low inclusion tier.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='section-header'>🧬 Development</div>",
                    unsafe_allow_html=True)
        hdi              = st.slider("HDI Score", 0.3, 0.95, 0.65, 0.01)
        health_index     = st.slider("Health Index", 0.3, 0.95, 0.70, 0.01)
        income_index     = st.slider("Income Index", 0.3, 0.95, 0.60, 0.01)
        education_index  = st.slider("Education Index", 0.2, 0.95, 0.55, 0.01)

    with col2:
        st.markdown("<div class='section-header'>💰 Wealth & Poverty</div>",
                    unsafe_allow_html=True)
        wealth_index       = st.slider("Wealth Index", 10.0, 80.0, 45.0, 0.5)
        poverty_rate       = st.slider("Poverty Rate (%)", 5.0, 90.0, 45.0, 0.5)
        electricity_access = st.slider("Electricity Access (%)", 20.0, 100.0, 80.0, 0.5)
        piped_water        = st.slider("Piped Water (%)", 1.0, 90.0, 40.0, 0.5)

    with col3:
        st.markdown("<div class='section-header'>📚 Education</div>",
                    unsafe_allow_html=True)
        avg_years_education  = st.slider("Avg Years Education", 1.0, 12.0, 5.0, 0.1)
        primary_attendance   = st.slider("Primary Attendance (%)", 30.0, 100.0, 80.0, 0.5)
        secondary_attendance = st.slider("Secondary Attendance (%)", 30.0, 100.0, 75.0, 0.5)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Financial Inclusion Score",
                 use_container_width=True):

        input_data = np.array([[
            hdi, health_index, income_index,
            education_index, wealth_index, poverty_rate,
            electricity_access, piped_water,
            avg_years_education, primary_attendance,
            secondary_attendance
        ]])

        fii_pred  = np.clip(rf_reg.predict(input_data)[0], 0, 1)
        tier_pred = le.inverse_transform(
            rf_clf.predict(input_data)
        )[0]
        tier_prob = rf_clf.predict_proba(input_data)[0]
        nat_mean  = master['fii_score'].mean()
        diff      = fii_pred - nat_mean

        tier_color_map = {
            'High':'#2ecc71','Medium':'#f39c12','Low':'#e74c3c'
        }
        tc = tier_color_map.get(tier_pred, '#ffffff')

        st.markdown(f"""
        <div style='background:#1a237e; border:2px solid {tc};
        border-radius:15px; padding:1.5rem; margin:1rem 0;
        text-align:center;'>
        <div style='font-size:1rem; color:#90caf9;
        margin-bottom:0.5rem;'>PREDICTED RESULT</div>
        <div style='font-size:3rem; font-weight:900;
        color:{tc};'>{fii_pred:.4f}</div>
        <div style='font-size:1.2rem; color:white;
        margin:0.3rem 0;'>Financial Inclusion Index Score</div>
        <div style='font-size:1.5rem; color:{tc};
        font-weight:700;'>Tier: {tier_pred} Inclusion</div>
        <div style='font-size:0.9rem; color:#90caf9;
        margin-top:0.5rem;'>
        vs National Average: {diff:+.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_p, col_d = st.columns(2)

        with col_p:
            st.markdown("<div class='section-header'>📊 Tier Probabilities</div>",
                        unsafe_allow_html=True)
            prob_df = pd.DataFrame({
                'Tier'       : le.classes_,
                'Probability': (tier_prob * 100).round(1)
            })
            fig_p = px.bar(
                prob_df, x='Tier', y='Probability',
                color='Tier',
                color_discrete_map={
                    'High'  :'#2ecc71',
                    'Low'   :'#e74c3c',
                    'Medium':'#f39c12'
                },
                text='Probability',
                height=300,
                labels={'Probability':'Probability (%)'}
            )
            fig_p.update_traces(texttemplate='%{text:.1f}%',
                                textposition='outside',
                                textfont_color='white')
            fig_p.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white', size=12),
                xaxis=dict(color='white',
                           tickfont=dict(color='white')),
                yaxis=dict(color='white',
                           tickfont=dict(color='white'),
                           gridcolor='#1f2937'),
                showlegend=False
            )
            st.plotly_chart(fig_p, use_container_width=True)

        with col_d:
            st.markdown("<div class='section-header'>🔑 Top Predictors</div>",
                        unsafe_allow_html=True)
            imp_df = pd.DataFrame({
                'Feature'   : features,
                'Importance': rf_reg.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)

            fig_i = px.bar(
                imp_df, x='Importance', y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues',
                height=300,
                labels={'Importance':'Importance Score',
                        'Feature':''}
            )
            fig_i.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white', size=12),
                xaxis=dict(color='white',
                           tickfont=dict(color='white'),
                           gridcolor='#1f2937'),
                yaxis=dict(color='white',
                           tickfont=dict(color='white')),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_i, use_container_width=True)

elif page == "🗂️ Cluster Explorer":

    st.markdown("""
    <div class='big-header'>
    <h1>🗂️ State Cluster Explorer</h1>
    <p>K-Means clustering of Indian states based on financial inclusion profile</p>
    </div>
    """, unsafe_allow_html=True)

    cluster_colors = {
        'Financially Advanced'  : '#2ecc71',
        'Developing Inclusion'  : '#3498db',
        'Emerging Inclusion'    : '#f39c12',
        'Financially Excluded'  : '#e74c3c'
    }

    col1, col2, col3, col4 = st.columns(4)
    for i, (cname, ccolor) in enumerate(cluster_colors.items()):
        count   = len(clusters[clusters['Cluster_Name']==cname])
        avg_fii = clusters[
            clusters['Cluster_Name']==cname
        ]['fii_score'].mean()
        cols    = [col1,col2,col3,col4]
        with cols[i]:
            st.markdown(f"""
            <div style='background:#0e1117;
            border:2px solid {ccolor};
            border-radius:12px; padding:1rem;
            text-align:center;'>
            <div style='color:{ccolor}; font-size:1rem;
            font-weight:800;'>{cname}</div>
            <div style='color:white; font-size:1.5rem;
            font-weight:900;'>{count}</div>
            <div style='color:#90caf9;
            font-size:0.85rem;'>States</div>
            <div style='color:{ccolor}; font-size:0.9rem;
            margin-top:0.3rem;'>Avg FII: {avg_fii:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🔵 Cluster Visualization (PCA)</div>",
                unsafe_allow_html=True)

    fig_cl = px.scatter(
        clusters,
        x='PCA1', y='PCA2',
        color='Cluster_Name',
        color_discrete_map=cluster_colors,
        text='State',
        height=580,
        labels={
            'PCA1'        :'Principal Component 1',
            'PCA2'        :'Principal Component 2',
            'Cluster_Name':'Cluster'
        }
    )
    fig_cl.update_traces(
        textposition='top center',
        textfont=dict(size=11, color='white'),
        marker=dict(size=14, opacity=0.9)
    )
    fig_cl.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white', size=12),
        xaxis=dict(gridcolor='#1f2937', color='white',
                   tickfont=dict(color='white'),
                   title_font=dict(color='white')),
        yaxis=dict(gridcolor='#1f2937', color='white',
                   tickfont=dict(color='white'),
                   title_font=dict(color='white')),
        legend=dict(font=dict(color='white', size=12),
                    bgcolor='#1a237e',
                    bordercolor='#3949ab'),
        title=dict(
            text='K-Means Clustering of Indian States',
            font=dict(color='white', size=15), x=0.5
        )
    )
    st.plotly_chart(fig_cl, use_container_width=True)

    st.markdown("<div class='section-header'>📋 States by Cluster</div>",
                unsafe_allow_html=True)

    for cname, ccolor in cluster_colors.items():
        with st.expander(f"  {cname}"):
            df_c = clusters[
                clusters['Cluster_Name']==cname
            ][['State','fii_score','hdi','poverty_rate']]\
             .sort_values('fii_score', ascending=False)\
             .reset_index(drop=True)
            st.dataframe(df_c, use_container_width=True)

elif page == "📊 Gap & TOPSIS Analysis":

    st.markdown("""
    <div class='big-header'>
    <h1>📊 Gap Analysis & TOPSIS Ranking</h1>
    <p>Identifying underperforming states and policy intervention priorities</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📉 Gap Analysis", "🏆 TOPSIS Ranking"])

    with tab1:
        st.markdown("""
        <div class='insight-box'>
        <b>Gap = Actual FII − Predicted FII</b><br><br>
        🔴 <b>Negative gap</b> = Underperforming
        — banking access below development potential<br>
        🟢 <b>Positive gap</b> = Overperforming
        — banking access exceeds development level<br>
        🔵 <b>Near zero</b> = On Track
        </div>
        """, unsafe_allow_html=True)

        gap_sorted = gap.sort_values('fii_gap')
        color_gap  = {
            'Underperforming':'#e74c3c',
            'On Track'       :'#3498db',
            'Overperforming' :'#2ecc71'
        }

        fig_g = px.bar(
            gap_sorted,
            x='fii_gap', y='State',
            orientation='h',
            color='performance',
            color_discrete_map=color_gap,
            height=750,
            labels={
                'fii_gap'    :'FII Gap (Actual − Predicted)',
                'State'      :'',
                'performance':'Performance'
            },
            text='fii_gap'
        )
        fig_g.update_traces(
            texttemplate='%{text:.3f}',
            textposition='outside',
            textfont_color='white'
        )
        fig_g.add_vline(
            x=0, line_dash='dash',
            line_color='white', line_width=2
        )
        fig_g.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white', size=12),
            xaxis=dict(gridcolor='#1f2937', color='white',
                       tickfont=dict(color='white'),
                       title_font=dict(color='white')),
            yaxis=dict(color='white',
                       tickfont=dict(color='white', size=11)),
            legend=dict(font=dict(color='white'),
                        bgcolor='#1a237e'),
            title=dict(
                text='Financial Inclusion Gap by State',
                font=dict(color='white', size=15), x=0.5
            )
        )
        st.plotly_chart(fig_g, use_container_width=True)

        st.dataframe(
            gap[['State','actual_fii','predicted_fii',
                 'fii_gap','performance',
                 'hdi','poverty_rate']]\
             .sort_values('fii_gap')\
             .reset_index(drop=True),
            use_container_width=True
        )

    with tab2:
        st.markdown("""
        <div class='insight-box'>
        <b>TOPSIS (Multi Criteria Decision Making)</b>
        ranks states by urgency of financial inclusion
        intervention using 6 weighted criteria:
        FII Score, HDI, Poverty Rate, Electricity Access,
        Education Years, and Wealth Index.
        </div>
        """, unsafe_allow_html=True)

        priority_colors = {
            'High Priority'  :'#e74c3c',
            'Medium Priority':'#f39c12',
            'Low Priority'   :'#2ecc71'
        }

        fig_t = px.bar(
            topsis.sort_values('topsis_score', ascending=True),
            x='topsis_score', y='State',
            orientation='h',
            color='priority_tier',
            color_discrete_map=priority_colors,
            height=750,
            text='topsis_rank',
            labels={
                'topsis_score' :'TOPSIS Score',
                'State'        :'',
                'priority_tier':'Priority'
            }
        )
        fig_t.update_traces(
            texttemplate='Rank #%{text}',
            textposition='outside',
            textfont_color='white'
        )
        fig_t.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white', size=12),
            xaxis=dict(gridcolor='#1f2937', color='white',
                       tickfont=dict(color='white'),
                       title_font=dict(color='white')),
            yaxis=dict(color='white',
                       tickfont=dict(color='white', size=11)),
            legend=dict(font=dict(color='white'),
                        bgcolor='#1a237e'),
            title=dict(
                text='TOPSIS Priority Ranking — Policy Intervention',
                font=dict(color='white', size=15), x=0.5
            )
        )
        st.plotly_chart(fig_t, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        for col, priority, color in zip(
            [col1, col2, col3],
            ['High Priority','Medium Priority','Low Priority'],
            ['#e74c3c','#f39c12','#2ecc71']
        ):
            with col:
                st.markdown(
                    f"<div class='section-header' "
                    f"style='color:{color};'>"
                    f"{priority}</div>",
                    unsafe_allow_html=True
                )
                df_p = topsis[
                    topsis['priority_tier']==priority
                ][['State','topsis_rank','poverty_rate']]
                st.dataframe(df_p, use_container_width=True)

elif page == "💡 Policy Recommendations":

    st.markdown("""
    <div class='big-header'>
    <h1>💡 Policy Recommendations</h1>
    <p>Data-driven insights for RBI, State Governments and NITI Aayog</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0a2e0a; border:2px solid #2ecc71;
    border-radius:12px; padding:1.5rem; margin:1rem 0;'>
    <div style='color:#2ecc71; font-size:1.2rem;
    font-weight:800; margin-bottom:0.8rem;'>
    🎯 Key Finding from Random Forest Model</div>
    <div style='color:white; font-size:1rem; line-height:1.8;'>
    Primary school attendance is the <b style='color:#2ecc71;'>
    #1 predictor of financial inclusion</b> with 29.9% importance
    — more than HDI, income, or wealth. States that invest in
    education today will have better banking access tomorrow.
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background:#1a237e; border:1px solid #3949ab;
        border-radius:12px; padding:1.5rem;'>
        <div style='color:#64b5f6; font-size:1.1rem;
        font-weight:800; margin-bottom:1rem;'>
        🏦 For RBI & Banking Sector</div>

        <div style='color:white; margin-bottom:1rem;'>
        <b style='color:#f39c12;'>1. Priority Branch Expansion</b><br>
        <span style='color:#e0e0e0;'>Focus new branch approvals in
        Bihar, Jharkhand, Assam and northeastern states —
        identified as High Priority by TOPSIS.</span>
        </div>

        <div style='color:white; margin-bottom:1rem;'>
        <b style='color:#f39c12;'>2. Business Correspondent Model</b><br>
        <span style='color:#e0e0e0;'>Strengthen BC network in states
        with poverty rates above 75% — UP, West Bengal, Odisha, MP.
        </span>
        </div>

        <div style='color:white; margin-bottom:1rem;'>
        <b style='color:#f39c12;'>3. Digital Banking Push</b><br>
        <span style='color:#e0e0e0;'>Punjab is underperforming
        relative to its development level — mobile banking
        and UPI adoption campaigns needed.</span>
        </div>

        <div style='color:white;'>
        <b style='color:#f39c12;'>4. Scale Successful Models</b><br>
        <span style='color:#e0e0e0;'>Rajasthan and UP show high
        credit-deposit ratios despite poverty —
        replicate to similar states.</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#1a237e; border:1px solid #3949ab;
        border-radius:12px; padding:1.5rem;'>
        <div style='color:#64b5f6; font-size:1.1rem;
        font-weight:800; margin-bottom:1rem;'>
        🏛️ For State Governments & NITI Aayog</div>

        <div style='color:white; margin-bottom:1rem;'>
        <b style='color:#f39c12;'>1. Education Investment</b><br>
        <span style='color:#e0e0e0;'>Primary attendance drives
        financial inclusion more than any other variable.
        Every 1% increase improves inclusion potential.</span>
        </div>

        <div style='color:white; margin-bottom:1rem;'>
        <b style='color:#f39c12;'>2. Infrastructure Development</b><br>
        <span style='color:#e0e0e0;'>Piped water and electricity
        are strong predictors — basic infrastructure
        precedes financial inclusion.</span>
        </div>

        <div style='color:white; margin-bottom:1rem;'>
        <b style='color:#f39c12;'>3. Northeast Special Package</b><br>
        <span style='color:#e0e0e0;'>Nagaland, Arunachal Pradesh,
        Manipur and Meghalaya need dedicated packages
        combining infrastructure, banking and education.
        </span>
        </div>

        <div style='color:white;'>
        <b style='color:#f39c12;'>4. Kerala Model Study</b><br>
        <span style='color:#e0e0e0;'>Kerala has high HDI (0.79)
        but medium FII — study barriers to formal banking
        despite high literacy.</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📊 Model Performance Summary</div>",
                unsafe_allow_html=True)

    perf_df = pd.DataFrame({
        'Model'    : [
            'Linear Regression','Ridge Regression',
            'Random Forest Regressor',
            'Logistic Regression',
            'Random Forest Classifier'
        ],
        'Type'     : [
            'Regression','Regression','Regression',
            'Classification','Classification'
        ],
        'Score'    : [
            'R² = 0.4366','R² = 0.3750','R² = 0.9047',
            'AUC = 0.8223','AUC = 0.9545'
        ],
        'Verdict'  : [
            'Baseline','Below Baseline','Best Model ⭐',
            'Good Baseline','Best Model ⭐'
        ]
    })
    st.dataframe(perf_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>⚠️ Limitations</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    • Dataset covers 2004–2014 (pre-UPI, pre-Jan Dhan maturity)<br>
    • Wealth and Education data averaged from 2006 and 2012 surveys<br>
    • FII is a composite index with equal component weightings<br>
    • State name standardization involved manual mapping decisions<br>
    • Results reflect structural patterns, not real-time banking status
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#90caf9;
    font-size:0.85rem; padding:1rem;
    border-top:1px solid #1565c0;'>
    MBA Applied Business Analytics Project<br>
    Data: RBI Handbook of Statistics on Indian States
    + Global Data Lab SHDI v10.1<br>
    Models: Random Forest (R²=0.90) + Logistic Regression
    (AUC=0.82) + TOPSIS MCDM
    </div>
    """, unsafe_allow_html=True)
