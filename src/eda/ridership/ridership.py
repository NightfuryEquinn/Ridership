#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Ridership Data
Objectives:
1. Temporal ridership trend analysis
2. Peak vs off-peak pattern profiling
3. Day-of-week & hour distribution
4. Growth rate & change-point detection

Note: Transit lines start reporting from their launch date.
Zero-filling is only applied AFTER the known launch date.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from ruptures.detection import Binseg

# Setup
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.max_columns = None

# Load data
def load_data(filepath):
    """Load and preprocess ridership data."""
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date')
    
    # Zero-fill missing values AFTER launch date for each service
    service_columns = [col for col in df.columns if col != 'total_ridership']
    for col in service_columns:
        # Extract launch date (first non-null value)
        launch_date = df[col].first_valid_index()
        if launch_date:
            # Zero-fill only after launch date
            df[col] = df[col].where(df.index < launch_date, df[col].fillna(0))
    
    return df

# Temporal Trend Analysis
def temporal_trend_analysis(df):
    """Analyze and visualize temporal trends."""
    plt.figure(figsize=(14, 8))
    
    # Plot total ridership
    plt.subplot(2, 1, 1)
    df['total_ridership'].plot(title='Total Ridership Over Time')
    plt.ylabel('Ridership')
    
    # Decompose trend, seasonality, and residuals
    decomposition = seasonal_decompose(df['total_ridership'].dropna(), model='additive', period=30)
    plt.subplot(2, 1, 2)
    decomposition.trend.plot(title='Trend Component')
    plt.ylabel('Trend')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_trend_analysis.png'))
    plt.close()
    
    # Save decomposition plots
    plt.figure(figsize=(14, 10))
    decomposition.plot()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ridership_decomposition.png'))
    plt.close()

# Peak vs Off-Peak Analysis
def peak_offpeak_analysis(df):
    """Profile peak vs off-peak patterns."""
    # Extract day of week and month
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_type'] = df['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    
    # Peak hours assumption (6-9 AM, 4-7 PM)
    df['hour'] = df.index.hour
    df['peak_period'] = df['hour'].apply(lambda x: 'Peak' if (6 <= x <= 9) or (16 <= x <= 19) else 'Off-Peak')
    
    # Aggregate by day type
    day_type_agg = df.groupby('day_type')['total_ridership'].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=day_type_agg, x='day_type', y='total_ridership')
    plt.title('Average Ridership: Weekday vs Weekend')
    plt.ylabel('Ridership')
    plt.savefig('peak_offpeak_day_type.png')
    plt.close()

# Day-of-Week and Hour Distribution
def day_hour_distribution(df):
    """Visualize day-of-week and hour distributions."""
    # Day of week
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    df['day_of_week'] = df.index.dayofweek
    sns.boxplot(data=df, x='day_of_week', y='total_ridership')
    plt.title('Ridership by Day of Week')
    plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.ylabel('Ridership')
    
    # Hour of day
    plt.subplot(1, 2, 2)
    df['hour'] = df.index.hour
    sns.boxplot(data=df, x='hour', y='total_ridership')
    plt.title('Ridership by Hour of Day')
    plt.ylabel('Ridership')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'peak_offpeak_day_type.png'))
    plt.close()

# Growth Rate and Change-Point Detection
def growth_changepoint_analysis(df):
    """Analyze growth rates and detect change points."""
    # Calculate monthly growth rate
    monthly_df = df['total_ridership'].resample('ME').sum()
    monthly_growth = monthly_df.pct_change() * 100
    
    plt.figure(figsize=(14, 6))
    monthly_growth.plot(title='Monthly Growth Rate (%)')
    plt.ylabel('Growth Rate (%)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'monthly_growth_rate.png'))
    plt.close()
    
    # Change-point detection using Binseg
    signal = df['total_ridership'].dropna().values
    n_samples = len(signal)
    model = Binseg(model="l2").fit(signal)
    change_points = model.predict(n_bkps=5)  # Top 5 change points
    
    # Filter valid change points
    change_points = [cp for cp in change_points if cp < n_samples]
    
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['total_ridership'], label='Ridership')
    for cp in change_points:
        if cp < len(df.index):
            plt.axvline(x=df.index[cp], color='r', linestyle='--', alpha=0.5)
    plt.title('Change-Point Detection in Ridership')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'changepoint_detection.png'))
    plt.close()

# Service-Specific Analysis
def service_specific_analysis(df):
    """Analyze trends for individual transit services."""
    service_columns = [col for col in df.columns if col not in ['total_ridership', 'day_of_week', 'month', 'day_type', 'hour', 'peak_period']]
    
    # Plot each service
    for service in service_columns:
        plt.figure(figsize=(12, 4))
        df[service].plot(title=f'{service} Ridership Over Time')
        plt.ylabel('Ridership')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'service_{service}.png'))
        plt.close()

# Main Execution
def main():
    """Run all EDA functions."""
    filepath = 'data/cleaned/ridership_headline_clean.csv'
    df = load_data(filepath)
    
    print("Running Temporal Trend Analysis...")
    temporal_trend_analysis(df)
    
    print("Running Peak vs Off-Peak Analysis...")
    peak_offpeak_analysis(df)
    
    print("Running Day-of-Week and Hour Distribution Analysis...")
    day_hour_distribution(df)
    
    print("Running Growth Rate and Change-Point Detection...")
    growth_changepoint_analysis(df)
    
    print("Running Service-Specific Analysis...")
    service_specific_analysis(df)
    
    print("EDA completed. Visualizations saved to current directory.")

if __name__ == "__main__":
    main()