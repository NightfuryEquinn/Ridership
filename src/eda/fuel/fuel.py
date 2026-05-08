import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# Load data
file_path = "../../../data/cleaned/fuelprice_cleaned.csv"
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'])

# Pivot data for level and change_weekly
level_data = data[data['series_type'] == 'level'].set_index('date').drop(columns=['series_type'])
change_data = data[data['series_type'] == 'change_weekly'].set_index('date').drop(columns=['series_type'])

# Create output directory if it doesn't exist
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# 1. Time-series trend and decomposition
def plot_decomposition(fuel_type, df):
    clean_data = df[fuel_type].dropna()
    period = 52 if len(clean_data) >= 104 else max(2, len(clean_data) // 4)

    plt.figure(figsize=(12, 8))
    decomposition = seasonal_decompose(clean_data, model='additive', period=period)
    decomposition.plot()
    plt.suptitle(f"Decomposition of {fuel_type} Price (RM)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"decomposition_{fuel_type}.png"))
    plt.close()

# 2. Price volatility & variance analysis
def plot_volatility(fuel_type, df):
    plt.figure(figsize=(12, 6))
    df[fuel_type].plot(title=f"Price Trend of {fuel_type} (RM)")
    plt.ylabel("Price (RM)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"trend_{fuel_type}.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    df[fuel_type].rolling(window=4).std().plot(title=f"Rolling Volatility (4-Week) of {fuel_type} Price")
    plt.ylabel("Volatility (RM)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"volatility_{fuel_type}.png"))
    plt.close()

# 3. Distribution & outlier detection
def plot_distribution(fuel_type, df):
    plt.figure(figsize=(12, 6))
    sns.histplot(df[fuel_type].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {fuel_type} Price (RM)")
    plt.xlabel("Price (RM)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"distribution_{fuel_type}.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df[fuel_type].dropna())
    plt.title(f"Boxplot of {fuel_type} Price (RM)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"boxplot_{fuel_type}.png"))
    plt.close()

# 4. Seasonal & cyclical pattern mining
def plot_seasonal_patterns(fuel_type, df):
    nlags = min(52, len(df[fuel_type].dropna()) // 2 - 1)
    if nlags < 10:
        nlags = 10

    plt.figure(figsize=(12, 6))
    plot_acf(df[fuel_type].dropna(), lags=nlags, title=f"Autocorrelation of {fuel_type} Price")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"acf_{fuel_type}.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plot_pacf(df[fuel_type].dropna(), lags=nlags, title=f"Partial Autocorrelation of {fuel_type} Price")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pacf_{fuel_type}.png"))
    plt.close()

    # Seasonal decomposition
    plot_decomposition(fuel_type, df)

# Generate plots for all fuel types
fuel_types = ['ron95', 'ron97', 'diesel', 'diesel_eastmsia', 'ron95_budi95', 'ron95_skps']

for fuel_type in fuel_types:
    if fuel_type in level_data.columns:
        plot_volatility(fuel_type, level_data)
        plot_distribution(fuel_type, level_data)
        plot_seasonal_patterns(fuel_type, level_data)
    if fuel_type in change_data.columns:
        plot_volatility(fuel_type, change_data)
        plot_distribution(fuel_type, change_data)

print("EDA plots generated successfully in the 'results' directory.")