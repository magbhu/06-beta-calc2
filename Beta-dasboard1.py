import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pathlib import Path

st.title("📈 Beta Calculator from NSE Historical Data")

# File uploads

default_stock_file = Path("stock.csv")
default_nifty_file = Path("niftybank.csv")

stock_file = default_stock_file.open("rb") if default_stock_file.exists() else  st.file_uploader("Upload Stock CSV", type="csv")
nifty_file = default_nifty_file.open("rb") if default_nifty_file.exists() else  st.file_uploader("Upload Nifty Index CSV", type="csv")

if stock_file and nifty_file:
    stock_df = pd.read_csv(stock_file, parse_dates=['Date'])
    index_df = pd.read_csv(nifty_file, parse_dates=['Date'])

    # Clean and prepare data
    stock_df = stock_df[['Date', 'Close']].rename(columns={'Close': 'Close_Stock'})
    index_df = index_df[['Date', 'Close']].rename(columns={'Close': 'Close_Nifty'})
    df = pd.merge(stock_df, index_df, on='Date')

    # Calculate returns
    df['Stock_Return'] = df['Close_Stock'].pct_change()
    df['Nifty_Return'] = df['Close_Nifty'].pct_change()
    df.dropna(inplace=True)

    # Run regression
    model = LinearRegression().fit(df[['Nifty_Return']], df['Stock_Return'])
    beta = model.coef_[0]

    st.subheader(f"📊 Calculated Beta: `{beta:.4f}`")

    # Plotting relationship
    st.write("### 🔍 Scatter Plot: Stock vs. Nifty Returns")
    fig, ax = plt.subplots()
    ax.scatter(df['Nifty_Return'], df['Stock_Return'], alpha=0.6)
    ax.plot(df['Nifty_Return'], model.predict(df[['Nifty_Return']]), color='red', label='Regression Line')
    ax.set_xlabel('Nifty Returns')
    ax.set_ylabel('Stock Returns')
    ax.set_title('Beta Estimation Scatter')
    ax.legend()
    st.pyplot(fig)
