import streamlit as st
from utils import *
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="InsightNAV | Mutual Fund Dashboard")


# --- Load saved data ---
bundle = joblib.load('tfidf_bundle.pkl')
pipeline = bundle['pipeline']
tfidf_matrix = bundle['tfidf_matrix']
schemes_df = bundle['schemes_df']

# --- Main UI Header ---
# ------------------ Header ------------------
st.markdown("## 📊 InsightNAV – Mutual Fund Performance Dashboard")

st.markdown("""
Welcome to **InsightNAV**, a tool for analyzing mutual fund performance against benchmarks.
This dashboard offers clear insights into fund returns, risk, and performance metrics using data visualizations and rolling CAGR comparisons.
""")

st.markdown("Use the search box below to select and analyze a mutual fund. The dashboard will display performance metrics, risk indicators, and comparisons with a benchmark.")

st.info("🔎 You can search using mutual fund names, AMC (fund house), or categories like 'large cap', 'mid cap', 'ELSS', etc.")

user_query = st.text_input(
    "🔎 Enter fund name, AMC, or keywords",
    placeholder="e.g. Axis Small Cap, SBI Bluechip, ELSS, large cap"
)

if user_query:
    user_query = clean_names(user_query)
    query_vec = pipeline.transform([user_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-10:][::-1]

    results = schemes_df.iloc[top_indices].copy()
    results.reset_index(drop=True, inplace=True)

    # --- Selection & Advanced Settings ---
    with st.form("selection_form"):
        st.markdown("### 🎯 Select a Scheme to Analyze")
        scheme_options = results["Scheme Name"].tolist()
        selected_option = st.selectbox("Choose from top results:", options=scheme_options)

        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.markdown("#### 🔧 Performance Calculation Parameters")
            col1, col2 = st.columns(2)

            with col1:
                risk_free_rate = st.slider( "📉 Risk-Free Annual Rate",
                    min_value=0.0,
                    max_value=0.15,
                    value=0.065,
                    step=0.001,
                    format="%.3f"
                )
                st.caption("Used in Sharpe Ratio, Alpha calculations")

            with col2:
                benchmark = st.selectbox("📊 Benchmark Index",
                    options=["Nifty 50", "Nifty Next 50", "Sensex"],
                    index=0
                )
                st.caption("Used to compare fund performance")

        submitted = st.form_submit_button("🔬 Analyze This Scheme")

    if submitted:
        selected_row = results[results["Scheme Name"] == selected_option].iloc[0]
        selected_code = selected_row["Scheme Code"]
        st.success(f"✅ You selected: **{selected_row['Scheme Name']}**")

        st.warning("""
        ⚠️ **Disclaimer:** This dashboard is for educational and informational purposes only.  
        It does **not constitute investment advice**. Please consult a SEBI-registered financial advisor before making any investment decisions.
        """)

        # ----------------------- Get The NAV and Index Data ------------------
        nav_df = get_nav(selected_code)
        index_data = get_index_data(benchmark, nav_df)

        # --------------------- Calculations ---------------------
        volatility = annualized_volatility(nav_df, 'Date', 'NAV')
        alpha_daily, alpha_annual, beta = calculate_alpha_beta(nav_df, index_data, risk_free_rate)
        sharpe = calculate_sharpe_ratio(nav_df, risk_free_rate)

        # Calculate
        merged_df = get_norm_data(nav_df, index_data)

        # --------------------- Plots ----------------------------
        fig = plot_two_lines(merged_df, x_col='Date', y_cols=['MF', 'Index'], title='MF vs Index', x_label='Date', y_label='Value')

        with st.expander("📘 How to Interpret These Metrics"):
            st.markdown("""
                        **Annual Volatility:** Shows price fluctuation over the past year.
                        **Beta:** Indicates sensitivity to market movements.    
                        **Alpha:** Measures outperformance relative to a benchmark.  
                        **Sharpe Ratio:** Indicates risk-adjusted return. A value above 1 is considered good.
            """)
        
        # Define a 4-column layout (25% + 75% split)
        grid_col = st.columns([1, 3])  # Column 1 = 25%, Columns 2-4 = 75%

        # ---------------- ROWS 1–3 ----------------

        # Left column (column 1): Metrics stacked vertically
        with grid_col[0]:
            st.markdown("### 📊 Key Metrics")
            st.metric("📉 Annual Volatility", f"{volatility:.2%}")
            st.metric("📊 Beta", f"{beta:.4f}")
            st.metric("📈 Annual Alpha", f"{alpha_annual:.2%}")
            st.metric("📌 Sharpe Ratio", f"{sharpe:.2f}")

        # Right side (columns 2–4 combined): NAV vs Benchmark plot (taking 3 rows visually)
        with grid_col[1]:
            st.markdown("### 📈 NAV vs Benchmark Over Time")
            st.plotly_chart(fig, use_container_width=True)

        # ---------------- ROW 4 and more ----------------
        st.markdown("### 🔁 Rolling CAGR: Mutual Fund vs Benchmark")

        with st.expander("ℹ️ What is Rolling CAGR?"):
            st.markdown("""
            Rolling CAGR tracks the **compound annual growth rate** over rolling time windows (1Y, 3Y, 5Y).  
            It helps investors evaluate **consistency** of returns rather than a single-point snapshot.
            """)
        
        if get_financial_year_span(nav_df, date_column='Date')>= 2:
            cagr_1 = merge_cagr_s(nav_df, index_data, period=1)
            fig_2 = plot_fund_vs_index(cagr_1, title='1-Year Rolling CAGR: Mutual Fund vs Benchmark')
            st.plotly_chart(fig_2, use_container_width=True)

        if get_financial_year_span(nav_df, date_column='Date')>= 4:
            cagr_3 = merge_cagr_s(nav_df, index_data, period=3)
            fig_3 = plot_fund_vs_index(cagr_3, title='3-Years Rolling CAGR: Mutual Fund vs Benchmark')
            st.plotly_chart(fig_3, use_container_width=True)

        if get_financial_year_span(nav_df, date_column='Date')>= 6:
            cagr_5 = merge_cagr_s(nav_df, index_data, period=5)
            fig_4 = plot_fund_vs_index(cagr_5, title='5-Years Rolling CAGR: Mutual Fund vs Benchmark')
            st.plotly_chart(fig_4, use_container_width=True)

        st.caption("📌 *Data sourced from publicly available APIs and resources. All outputs are for personal, non-commercial use only.*")

st.markdown("---")  # Horizontal divider

# About and Footer
st.markdown("""
👨‍💻 Built by [Saarthak Jain](https://saarthakjain.vercel.app/)  
🔗 [GitHub](https://github.com/SaarthakJain01) | [LinkedIn](https://www.linkedin.com/in/saarthakjain01)
""")