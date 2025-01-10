# app.py

from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from CorporateBondPortfolioOptimizer import CorporateBondPortfolioOptimizer
from GovernmentBondPortfolioOptimizer import GovernmentBondPortfolioOptimizer
from MPT import MPT

# Set page configuration
st.set_page_config(page_title="Optimasi Portofolio Obligasi", layout="wide", initial_sidebar_state="expanded")

# Sidebar for navigation
st.sidebar.title("Navigasi")
app_mode = st.sidebar.selectbox("Pilih Mode Aplikasi", ["Portofolio MPT Awal", "Optimasi Obligasi Korporasi", "Optimasi Obligasi Pemerintah"])


# MPT Portfolio Awal
if app_mode == "Portofolio MPT Awal":
    # Configure the Streamlit app
    # st.set_page_config(page_title="MPT Portfolio Optimization", layout="wide")

    # Title and description
    st.title("Optimisasi Portofolio Teori Portofolio Modern (MPT)")
    st.write("Aplikasi ini melakukan optimisasi portofolio dengan menggunakan Teori Portofolio Modern (MPT).")

    # ---------------------
    # Step 1: Input Parameters
    # ---------------------
    st.sidebar.header("Parameter Masukan")

    # Input for tickers
    tickers_input = st.sidebar.text_input(
        "Masukkan simbol ticker yang dipisahkan dengan koma",
        'BBCA.JK, BBRI.JK, BMRI.JK, TLKM.JK, ASII.JK, BBNI.JK, UNTR.JK, ADRO.JK, ICBP.JK, INDF.JK'
    )
    tickers = [tick.strip() for tick in tickers_input.split(',')]

    # Input for date range
    start_date = st.sidebar.date_input("Tanggal Mulai", pd.to_datetime("2021-12-17"))
    end_date = st.sidebar.date_input("Tanggal Selesai", pd.to_datetime("2024-12-17"))

    # Input for risk-free rate
    risk_free = st.sidebar.number_input("Tingkat Bebas Risiko (tahunan)", min_value=0.0, max_value=1.0, value=0.07)

    # Upload IBPREXTR data
    uploaded_file = st.sidebar.file_uploader("Unggah File Excel IBPREXTR", type=["xlsx"])

    if uploaded_file is not None:
        # Read ibprextr_data
        ibprextr_data = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
        ibprextr_data.index = pd.to_datetime(ibprextr_data.index)

        # ---------------------
        # Step 2: Data Retrieval and Processing
        # ---------------------
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        data.index = pd.to_datetime(data.index)
        data = data.join(ibprextr_data, how='inner')
        data = data.dropna(axis=1)
        data.columns = data.columns.astype(str)
        returns = data.pct_change().dropna()

        # Create an instance of the MPT class
        mpt = MPT(returns, risk_free_rate=risk_free)

        # ---------------------
        # Step 3: Optimization and Efficient Frontier
        # ---------------------
        # Optimize portfolios
        tan_result = mpt.maximize_sharpe()
        min_vol_result = mpt.minimize_volatility()

        tan_weights = tan_result.x
        min_vol_weights = min_vol_result.x

        tan_ret, tan_vol = mpt.portfolio_performance(tan_weights)
        min_vol_ret, min_vol_vol = mpt.portfolio_performance(min_vol_weights)

        tan_sharpe = mpt.sharpe_ratio(tan_ret, tan_vol)
        min_vol_sharpe = mpt.sharpe_ratio(min_vol_ret, min_vol_vol)

        # Calculate Efficient Frontier
        frontier_volatilities, target_returns = mpt.calculate_efficient_frontier()

        # ---------------------
        # Step 4: Display Results
        # ---------------------
        st.header("Frontier Efisien")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frontier_volatilities, target_returns, 'b--', label='Efficient Frontier')
        ax.scatter(min_vol_vol, min_vol_ret, marker='*', color='r', s=200, label='Minimum Volatility')
        ax.scatter(tan_vol, tan_ret, marker='*', color='g', s=200, label='Tangency Portfolio')
        ax.set_xlabel('Volatilitas Tahunan')
        ax.set_ylabel('Return Tahunan')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Portfolio Allocations
        labels = mpt.assets

        st.header("Alokasi Portofolio")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Portofolio Tangensi")
            tan_weights_rounded = np.round(tan_weights * 100, 2)
            fig_tan = go.Figure(data=[go.Pie(labels=labels, values=tan_weights_rounded, hole=.3)])
            fig_tan.update_layout(title_text=f'Return: {tan_ret*100:.2f}% | Volatility: {tan_vol*100:.2f}% | Sharpe Ratio: {tan_sharpe:.2f}')
            st.plotly_chart(fig_tan)

        with col2:
            st.subheader("Portofolio Volatilitas Minimum")
            min_vol_weights_rounded = np.round(min_vol_weights * 100, 2)
            fig_min = go.Figure(data=[go.Pie(labels=labels, values=min_vol_weights_rounded, hole=.3)])
            fig_min.update_layout(title_text=f'Return: {min_vol_ret*100:.2f}% | Volatility: {min_vol_vol*100:.2f}% | Sharpe Ratio: {min_vol_sharpe:.2f}')
            st.plotly_chart(fig_min)

        st.subheader("Bobot Portofolio Tangensi")
        tan_weights_df = pd.DataFrame({'Asset': labels, 'Weight (%)': tan_weights_rounded})
        st.table(tan_weights_df)

        st.subheader("Bobot Portofolio Volatilitas Minimum")
        min_vol_weights_df = pd.DataFrame({'Asset': labels, 'Weight (%)': min_vol_weights_rounded})
        st.table(min_vol_weights_df)

    else:
        st.info("Silakan unggah file Excel IBPREXTR untuk melanjutkan.")

# Corporate Bond Optimizer
elif app_mode == "Optimasi Obligasi Korporasi":
    # Set page configuration
    # st.set_page_config(
    #     page_title="Bond Portfolio Optimizer (Corporate Edition)",
    #     layout="wide",
    #     initial_sidebar_state="expanded",
    # )

    # Title and Description
    st.title("Optimasi Portofolio Obligasi (Edisi Korporasi)")

    st.subheader("Model Matematika")
    st.write(
        """
        Optimisasi di aplikasi ini didasarkan pada **Optimisasi Konveks**, 
        yang bertujuan memaksimalkan ekspektasi imbal hasil portofolio obligasi 
        sambil memenuhi kendala yang ditentukan.
        """
    )

    # Objective Function
    st.subheader("Fungsi Objektif")
    st.latex(r"Maksimalkan \ R_{p}")
    st.write("Dimana:")
    st.write(
        """
        - **Rₚ**: Expected portfolio return, calculated as:
        """
    )
    st.latex(r"R_{p} = \mathbf{w}^\top \mathbf{r}")
    st.write(
        """
        - **w**: Weight vector of portfolio allocation.
        - **r**: Vector of individual bond yields.
        """
    )

    # Membuat dua kolom: satu untuk Constraints dan satu untuk Rating Scores
    col1, col2 = st.columns([3, 1.5])

    with col1:
        # Constraints
        st.subheader("Kendala")
        st.latex(
            r"""
            \begin{aligned}
            1. & \quad \sum_{i} w_{i} = 1 \quad \text{(Total portfolio weight must equal 1)} \\
            2. & \quad w_{i} \geq 0 \quad \text{(No short selling allowed)} \\
            3. & \quad w_{i} \leq \text{Max Position Size} \quad \text{(Limit on maximum exposure to any single bond)} \\
            4. & \quad R_{p} \geq \text{Min Return} \quad \text{(Minimum portfolio return constraint)} \\
            5. & \quad \mathbf{w}^\top \mathbf{q} \geq \text{Min Rating Score} \quad \text{(Minimum average rating score for the portfolio)} \\
            \end{aligned}
            """
        )

    with col2:
        # PEFINDO Rating Scores
        st.subheader("Skor Peringkat PEFINDO")
        rating_scores = {
            'idAAA': 1.00, 'idAA+': 0.95, 'idAA': 0.90, 'idAA-': 0.85,
            'idA+': 0.80, 'idA': 0.75, 'idA-': 0.70,
            'idBBB+': 0.65, 'idBBB': 0.60, 'idBBB-': 0.55,
            'idBB+': 0.50, 'idBB': 0.45, 'idBB-': 0.40,
            'idB+': 0.35, 'idB': 0.30, 'idB-': 0.25
        }

        # Buat DataFrame dari rating_scores
        rating_df = pd.DataFrame(list(rating_scores.items()), columns=['PEFINDO Rating', 'Score'])

        # Tampilkan tabel dengan format dua desimal dan ukuran yang lebih besar
        st.dataframe(
            rating_df.set_index('PEFINDO Rating').style.format({'Score': "{:.2f}"}),
            height=400  # Menentukan tinggi tabel agar lebih besar
        )

    # Sidebar for Data Upload
    st.sidebar.header("1. Unggah File Data")

    uploaded_bonds = st.sidebar.file_uploader("Upload Bonds.xlsx", type=["xlsx"])

    if uploaded_bonds:
        # Read the uploaded Excel file
        bonds_df = pd.read_excel(uploaded_bonds)

        # Check if 'Series' column exists
        if 'Series' not in bonds_df.columns:
            st.sidebar.error("The uploaded Bonds.xlsx file must contain a 'Series' column.")
            st.stop()

        st.sidebar.success("Bonds.xlsx berhasil diunggah!")
    else:
        st.sidebar.warning("Harap unggah file Bonds.xlsx untuk melanjutkan.")
        st.stop()

    # Sidebar for Optimization Parameters
    st.sidebar.header("2. Atur Parameter Optimisasi")

    total_investment = st.sidebar.number_input(
        "Total Investasi (Rp)", 
        min_value=1000.0, 
        max_value=1e12,  # Increased max value for flexibility
        value=1000000.0, 
        step=1000.0
    )

    min_return = st.sidebar.number_input(
        "Minimum Return yang Diharapkan (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=7.0, 
        step=0.1
    ) / 100  # Convert to decimal

    max_position_size = st.sidebar.slider(
        "Batasan Posisi Maksimum (%)", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.3, 
        step=0.01
    )

    rating_constraint_type = st.sidebar.radio(
        "Jenis Kendala Peringkat",
        ["Rata-Rata Peringkat Portofolio", "Peringkat Minimal Individu"],
        help="Pilih apakah kendala peringkat diterapkan di level portofolio atau obligasi individu"
    )

    if rating_constraint_type == "Rata-Rata Peringkat Portofolio":
        aggregate_rating_score = st.sidebar.slider(
            "Skor Rata-Rata Peringkat Portofolio", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.05,
            help="Target skor rata-rata peringkat untuk seluruh portofolio"
        )
        min_rating_score = None
    else:  # Minimum Individual Rating
        min_rating_score = st.sidebar.slider(
            "Skor Peringkat Minimal Individu", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.05,
            help="Skor peringkat minimal yang diperbolehkan untuk setiap obligasi individu"
        )
        aggregate_rating_score = None

    # Button to Optimize
    st.sidebar.header("3. Jalankan Optimisasi")
    optimize_button = st.sidebar.button("Optimalkan Portofolio")

    if optimize_button:
        with st.spinner("Mengoptimalkan portofolio..."):
            # Initialize the optimizer
            optimizer = CorporateBondPortfolioOptimizer(
                bonds_df=bonds_df
            )

            # Optimize the portfolio
            weights, results = optimizer.optimize_portfolio(
                total_investment=total_investment,
                min_return=min_return,
                max_position_size=max_position_size,
                aggregate_rating_score=aggregate_rating_score,
                min_rating_score=min_rating_score
            )

            # Generate report
            portfolio_data, summary_data, figures = optimizer.generate_report(weights)

        # Display Optimization Results
        st.success("Portofolio berhasil dioptimalkan!")

        # Display Summary Metrics
        st.subheader("📊 Ringkasan Portofolio")
        summary_df = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
        summary_df = summary_df[summary_df['Metric'] != 'Analysis Date']  # Exclude Analysis Date
        st.table(summary_df.set_index('Metric'))

        # Display Detailed Portfolio Allocation
        st.subheader("📈 Alokasi Portofolio")
        # Filter out bonds with negligible weights
        allocation_df = portfolio_data[portfolio_data['Weight (%)'] > 0.01].copy()
        # Convert Investment to Rupiah
        allocation_df['Investment (Rp)'] = allocation_df['Investment (Rp)']
        allocation_df = allocation_df[[
            'Issuer', 'Ticker', 'Series', 'Weight (%)', 'Yield (%)',  # Ensure 'Yield (%)' uses Yld to Mty (Mid)
            'Duration (years)', 'Rating', 'Investment (Rp)'
        ]]
        allocation_df.rename(columns={
            'Issuer': 'Issuer',
            'Ticker': 'Ticker',
            'Series': 'Series',
            'Weight (%)': 'Weight (%)',
            'Yield (%)': 'Yld to Mty (Mid)',  # Rename for clarity
            'Duration (years)': 'Duration (years)',
            'Rating': 'Rating',
            'Investment (Rp)': 'Investment (Rp)'
        }, inplace=True)
        st.dataframe(allocation_df.style.format({
            'Weight (%)': "{:.2f}",
            'Yld to Mty (Mid)': "{:.2f}",
            'Duration (years)': "{:.2f}",
            'Investment (Rp)': "{:,.0f}"
        }))

        # Tambahkan Expected Portfolio Return di Bawah Tabel Alokasi
        st.subheader("📈 Return Portofolio yang Diharapkan")
        st.metric(
            label="Expected Portfolio Return (%)",
            value=f"{summary_data['Portfolio Yield (%)']}%"
        )

        # Display Interactive Plots
        st.subheader("📊 Visualisasi Interaktif")

        # Portfolio Composition Sunburst
        st.plotly_chart(figures['composition'], use_container_width=True)

        # Rating Distribution Treemap
        st.plotly_chart(figures['ratings'], use_container_width=True)

        # The following sections have been removed:
        # - 💾 Download Interactive Report
        # - ✅ Optimization Details

    else:
        st.info("Unggah file data Anda dan atur parameter optimisasi, lalu klik 'Optimalkan Portofolio' untuk memulai.")


# Government Bond Optimizer
elif app_mode == "Optimasi Obligasi Pemerintah":
    # Set page configuration
    # st.set_page_config(
    #     page_title="Government Bond Portfolio Optimizer",
    #     layout="wide",
    #     initial_sidebar_state="expanded",
    # )

    # Title and Description
    st.title("Optimasi Portofolio Obligasi Pemerintah")

    st.subheader("Model Matematika")
    st.write(
        """
        Optimisasi di aplikasi ini didasarkan pada **Optimisasi Konveks**, 
        yang bertujuan memaksimalkan ekspektasi imbal hasil portofolio obligasi pemerintah 
        sambil memenuhi kendala yang ditentukan.
        """
    )

    # Objective Function
    st.subheader("Fungsi Objektif")
    st.latex(r"Maksimalkan \ R_{p}")
    st.write("Dimana:")
    st.write(
        """
        - **Rₚ**: Expected portfolio return, calculated as:
        """
    )
    st.latex(r"R_{p} = \mathbf{w}^\top \mathbf{r}")
    st.write(
        """
        - **\(\mathbf{w}\)**: Weight vector of portfolio allocation.
        - **\(\mathbf{r}\)**: Vector of individual bond yields.
        """
    )

    # Membuat dua kolom: satu untuk Constraints dan satu (dulunya untuk Rating Scores)
    col1, col2 = st.columns([3, 1.5])

    with col1:
        # Constraints
        st.subheader("Kendala")
        st.latex(
            r"""
            \begin{aligned}
            1. & \quad \sum_{i} w_{i} = 1 \quad \text{(Total portfolio weight must equal 1)} \\
            2. & \quad w_{i} \geq 0 \quad \text{(No short selling allowed)} \\
            3. & \quad w_{i} \leq \text{Max Position Size} \quad \text{(Limit on maximum exposure to any single bond)} \\
            4. & \quad R_{p} \geq \text{Min Return} \quad \text{(Minimum portfolio return constraint)} \\
            \end{aligned}
            """
        )

    # Bagian "Note" tentang PEFINDO dihapus
    # with col2:
    #     # Bagian ini dihapus sesuai permintaan
    #     pass

    # Sidebar for Data Upload
    st.sidebar.header("1. Unggah File Data")

    uploaded_bonds = st.sidebar.file_uploader("Upload Bonds.xlsx", type=["xlsx"])

    if uploaded_bonds:
        # Read the uploaded Excel file
        bonds_df = pd.read_excel(uploaded_bonds)

        # Check if required columns exist
        required_columns = ['Issuer Name', 'Ticker', 'Yld to Mty (Mid)', 'Cpn', 'Maturity', 'Series', 'Currency']
        if not all(col in bonds_df.columns for col in required_columns):
            st.sidebar.error(f"The uploaded Bonds.xlsx file must contain the following columns: {', '.join(required_columns)}.")
            st.stop()

        st.sidebar.success("Bonds.xlsx berhasil diunggah!")
    else:
        st.sidebar.warning("Harap unggah file Bonds.xlsx untuk melanjutkan.")
        st.stop()

    # Sidebar for Optimization Parameters
    st.sidebar.header("2. Atur Parameter Optimisasi")

    total_investment = st.sidebar.number_input(
        "Total Investasi (Rp)", 
        min_value=1000.0, 
        max_value=1e12,  # Increased max value for flexibility
        value=1000000.0, 
        step=1000.0
    )

    min_return = st.sidebar.number_input(
        "Minimum Return yang Diharapkan (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=7.0, 
        step=0.1
    ) / 100  # Convert to decimal

    max_position_size = st.sidebar.slider(
        "Batasan Posisi Maksimum (%)", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.3, 
        step=0.01
    )

    # Button to Optimize
    st.sidebar.header("3. Jalankan Optimisasi")
    optimize_button = st.sidebar.button("Optimalkan Portofolio")

    if optimize_button:
        with st.spinner("Mengoptimalkan portofolio..."):
            # Initialize the optimizer
            optimizer = GovernmentBondPortfolioOptimizer(
                bonds_df=bonds_df
            )

            # Optimize the portfolio
            weights, results = optimizer.optimize_portfolio(
                total_investment=total_investment,
                min_return=min_return,
                max_position_size=max_position_size
            )

            # Generate report
            portfolio_data, summary_data, figures = optimizer.generate_report(weights)

        # Display Optimization Results
        st.success("Portofolio berhasil dioptimalkan!")

        # Display Summary Metrics
        st.subheader("📊 Ringkasan Portofolio")
        summary_df = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
        summary_df = summary_df[summary_df['Metric'] != 'Analysis Date']  # Exclude Analysis Date
        st.table(summary_df.set_index('Metric'))

        # Display Detailed Portfolio Allocation
        st.subheader("📈 Alokasi Portofolio")
        # Filter out bonds with negligible weights
        allocation_df = portfolio_data[portfolio_data['Weight (%)'] > 0.01].copy()
        allocation_df = allocation_df[[
            'Issuer', 'Ticker', 'Series', 'Weight (%)', 'Yield (%)',
            'Duration (years)', 'Maturity', 'Investment (Rp)'
        ]]
        allocation_df.rename(columns={
            'Issuer': 'Issuer',
            'Ticker': 'Ticker',
            'Series': 'Series',
            'Weight (%)': 'Weight (%)',
            'Yield (%)': 'Yield (%)',
            'Duration (years)': 'Duration (years)',
            'Maturity': 'Maturity',
            'Investment (Rp)': 'Investment (Rp)'
        }, inplace=True)
        st.dataframe(allocation_df.style.format({
            'Weight (%)': "{:.2f}",
            'Yield (%)': "{:.2f}",
            'Duration (years)': "{:.2f}",
            'Investment (Rp)': "{:,.0f}"
        }))

        # Tampilkan Expected Portfolio Return di bawah tabel
        st.subheader("📈 Return Portofolio yang Diharapkan")
        st.metric(
            label="Expected Portfolio Return (%)",
            value=f"{summary_data['Portfolio Yield (%)']}%"
        )

        # Display Interactive Plots
        st.subheader("📊 Visualisasi Interaktif")

        # Portfolio Composition Sunburst
        st.plotly_chart(figures['composition'], use_container_width=True)

        # Duration Distribution Bar Chart
        st.plotly_chart(figures['duration'], use_container_width=True)

    else:
        st.info("Unggah file data Anda dan atur parameter optimisasi, lalu klik 'Optimalkan Portofolio' untuk memulai.")
