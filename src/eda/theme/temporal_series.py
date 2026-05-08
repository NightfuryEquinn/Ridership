"""
Temporal/Time-Series Theme EDA
Fuel price trends, ridership time series, rainfall seasonality, holiday calendar effects,
change-point detection across all temporal datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import json
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "cleaned"
OUTPUT_DIR = BASE_DIR / "src" / "eda" / "theme" / "results"

GRAPH_DIR = OUTPUT_DIR / "graphs"
CSV_DIR = OUTPUT_DIR / "csv"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def load_all_data():
    """Load all temporal datasets."""
    fuel = pd.read_csv(DATA_DIR / "fuelprice_cleaned.csv", parse_dates=['date'])
    rainfall = pd.read_csv(DATA_DIR / "rainfall_combined_final.csv", parse_dates=['date'])
    ridership = pd.read_csv(DATA_DIR / "ridership_headline_clean.csv", parse_dates=['date'])
    holidays = pd.read_csv(DATA_DIR / "school_public_holiday_clean.csv", parse_dates=['start_date', 'end_date'])
    return fuel, rainfall, ridership, holidays


def plot_fuel_price_trends(fuel):
    """Fuel price trends over time - levels and weekly changes."""
    fuel_levels = fuel[fuel['series_type'] == 'level'].copy()
    fuel_changes = fuel[fuel['series_type'] == 'change_weekly'].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    ax1.plot(fuel_levels['date'], fuel_levels['ron95'], label='RON95', color='#e74c3c', linewidth=1.5)
    ax1.plot(fuel_levels['date'], fuel_levels['ron97'], label='RON97', color='#3498db', linewidth=1.5)
    ax1.plot(fuel_levels['date'], fuel_levels['diesel'], label='Diesel', color='#2ecc71', linewidth=1.5)
    ax1.set_title('Fuel Price Levels (MYR/Litre)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (MYR)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax2 = axes[1]
    ax2.bar(fuel_changes['date'], fuel_changes['ron95'], label='RON95', color='#e74c3c', alpha=0.7, width=5)
    ax2.bar(fuel_changes['date'], fuel_changes['diesel'], label='Diesel', color='#2ecc71', alpha=0.7, width=5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Weekly Fuel Price Changes', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Change (MYR)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'fuel_price_trends.png', dpi=150, bbox_inches='tight')
    plt.close()

    summary = fuel_levels.groupby(fuel_levels['date'].dt.year).agg({
        'ron95': ['mean', 'min', 'max'],
        'diesel': ['mean', 'min', 'max']
    }).round(3)
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary.to_csv(CSV_DIR / 'fuel_price_summary.csv')


def plot_ridership_timeseries(ridership):
    """Ridership time series for all transit modes."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    modes = ['bus_rkl', 'bus_rpn', 'rail_lrt_ampang', 'rail_mrt_kajang', 'rail_lrt_kj', 'rail_monorail', 'rail_mrt_pjy']
    mode_names = ['Bus RKL', 'Bus RPN', 'LRT Ampang', 'MRT Kajang', 'LRT Kelana Jaya', 'Monorail', 'MRT PJY']
    colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))

    ax1 = axes[0]
    for mode, name, color in zip(modes, mode_names, colors):
        if mode in ridership.columns:
            valid_data = ridership[['date', mode]].dropna()
            if len(valid_data) > 0:
                ax1.plot(valid_data['date'], valid_data[mode], label=name, linewidth=1, alpha=0.8)

    ax1.set_title('Daily Ridership by Transit Mode', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Daily Ridership')
    ax1.legend(loc='upper left', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax2 = axes[1]
    ax2.plot(ridership['date'], ridership['total_ridership'], color='#2c3e50', linewidth=1.5)
    ax2.fill_between(ridership['date'], ridership['total_ridership'], alpha=0.3, color='#3498db')
    ax2.set_title('Total Daily Ridership (All Modes)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Total Ridership')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'ridership_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()

    ridership_summary = ridership.copy()
    ridership_summary['year_month'] = ridership_summary['date'].dt.to_period('M')
    monthly = ridership_summary.groupby('year_month').agg({
        'total_ridership': ['mean', 'sum', 'min', 'max']
    }).round(0)
    monthly.columns = ['avg_daily', 'total_monthly', 'min_daily', 'max_daily']
    monthly.to_csv(CSV_DIR / 'ridership_monthly_summary.csv')


def plot_rainfall_seasonality(rainfall):
    """Rainfall seasonality analysis."""
    rainfall_nat = rainfall.copy()
    rainfall_nat['date'] = pd.to_datetime(rainfall_nat['date'])
    rainfall_nat['month'] = rainfall_nat['date'].dt.month
    rainfall_nat['year'] = rainfall_nat['date'].dt.year

    rainfall_level1 = rainfall_nat[rainfall_nat['adm_level'] == 1].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    monthly_avg = rainfall_level1.groupby('month')['rainfall_mm'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    colors = plt.cm.Blues(monthly_avg / monthly_avg.max())
    ax1.bar(range(1, 13), monthly_avg.values, color=colors)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(months)
    ax1.set_title('Average Daily Rainfall by Month', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Rainfall (mm)')

    ax2 = axes[0, 1]
    anomaly_by_month = rainfall_level1.groupby('month')['anomaly_rf'].mean()
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in anomaly_by_month.values]
    ax2.bar(range(1, 13), anomaly_by_month.values, color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(months)
    ax2.set_title('Rainfall Anomaly by Month (% from average)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Anomaly (%)')

    ax3 = axes[1, 0]
    sample_state = rainfall_level1[rainfall_level1['NAME_1'] == rainfall_level1['NAME_1'].iloc[0]] if 'NAME_1' in rainfall_level1.columns else rainfall_level1.head(500)
    if 'NAME_1' in rainfall_level1.columns:
        sample_state = rainfall_level1[rainfall_level1['NAME_1'].isin(rainfall_level1['NAME_1'].unique()[:3])]
        for state in sample_state['NAME_1'].unique():
            state_data = sample_state[sample_state['NAME_1'] == state].sort_values('date')
            ax3.plot(state_data['date'], state_data['rainfall_mm'].rolling(7).mean(), label=state, linewidth=1)
    else:
        ax3.plot(rainfall_level1['date'].head(500), rainfall_level1['rainfall_mm'].head(500).rolling(7).mean(), linewidth=1)
    ax3.set_title('7-Day Rolling Average Rainfall', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Rainfall (mm)')
    ax3.legend(loc='upper right')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax4 = axes[1, 1]
    yearly_totals = rainfall_level1.groupby('year')['rainfall_mm'].sum()
    ax4.bar(yearly_totals.index, yearly_totals.values, color='#3498db', alpha=0.8)
    ax4.set_title('Total Annual Rainfall', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Total Rainfall (mm)')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'rainfall_seasonality.png', dpi=150, bbox_inches='tight')
    plt.close()

    seasonal_summary = rainfall_level1.groupby('month').agg({
        'rainfall_mm': ['mean', 'std'],
        'anomaly_rf': 'mean'
    }).round(2)
    seasonal_summary.columns = ['rainfall_mean', 'rainfall_std', 'anomaly_mean']
    seasonal_summary.to_csv(CSV_DIR / 'rainfall_seasonal_summary.csv')


def plot_holiday_calendar_effects(ridership, holidays):
    """Holiday calendar effects on ridership."""
    holidays['start_date'] = pd.to_datetime(holidays['start_date'])
    holidays['end_date'] = pd.to_datetime(holidays['end_date'])

    ridership = ridership.copy()
    ridership['date'] = pd.to_datetime(ridership['date'])
    ridership['dayofweek'] = ridership['date'].dt.dayofweek
    ridership['month'] = ridership['date'].dt.month
    ridership['is_weekend'] = ridership['dayofweek'].isin([5, 6])

    holiday_dates = set()
    for _, row in holidays.iterrows():
        date_range = pd.date_range(start=row['start_date'], end=row['end_date'])
        holiday_dates.update(date_range)

    ridership['is_holiday'] = ridership['date'].isin(holiday_dates)
    ridership['day_type'] = 'Normal'
    ridership.loc[ridership['is_weekend'], 'day_type'] = 'Weekend'
    ridership.loc[ridership['is_holiday'], 'day_type'] = 'Holiday'
    ridership.loc[ridership['is_weekend'] & ridership['is_holiday'], 'day_type'] = 'Weekend+Holiday'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    day_type_stats = ridership.groupby('day_type')['total_ridership'].agg(['mean', 'std', 'count']).round(0)

    ax1 = axes[0, 0]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax1.bar(day_type_stats.index, day_type_stats['mean'], yerr=day_type_stats['std'],
                   color=colors[:len(day_type_stats)], capsize=5)
    ax1.set_title('Average Ridership by Day Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Daily Ridership')
    for bar, count in zip(bars, day_type_stats['count']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f'n={int(count)}', ha='center', va='bottom', fontsize=9)

    ax2 = axes[0, 1]
    public_holidays = holidays[holidays['type'] == 'Public']
    holiday_by_month = public_holidays.groupby(public_holidays['start_date'].dt.month).size()
    ax2.bar(holiday_by_month.index, holiday_by_month.values, color='#9b59b6')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax2.set_title('Public Holidays by Month', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Number of Holidays')

    ax3 = axes[1, 0]
    dayofweek_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_stats = ridership.groupby('dayofweek')['total_ridership'].mean()
    colors_dow = ['#3498db' if i < 5 else '#e74c3c' for i in range(7)]
    ax3.bar(range(7), dow_stats.values, color=colors_dow)
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(dayofweek_names)
    ax3.set_title('Average Ridership by Day of Week', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Day of Week')
    ax3.set_ylabel('Average Daily Ridership')

    ax4 = axes[1, 1]
    monthly_ridership = ridership.groupby('month')['total_ridership'].mean()
    ax4.plot(range(1, 13), monthly_ridership.values, marker='o', linewidth=2, color='#2c3e50')
    ax4.fill_between(range(1, 13), monthly_ridership.values, alpha=0.3, color='#3498db')
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax4.set_title('Average Ridership by Month', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Average Daily Ridership')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'holiday_calendar_effects.png', dpi=150, bbox_inches='tight')
    plt.close()

    daytype_summary = ridership.groupby(['day_type', 'month'])['total_ridership'].mean().unstack(fill_value=0)
    daytype_summary.to_csv(CSV_DIR / 'ridership_by_daytype_month.csv')


def detect_changepoints(ridership, fuel):
    """Simple change-point detection using rolling statistics."""
    ridership = ridership.copy()
    ridership['date'] = pd.to_datetime(ridership['date'])
    ridership = ridership.sort_values('date')

    fuel_levels = fuel[fuel['series_type'] == 'level'].copy()
    fuel_levels['date'] = pd.to_datetime(fuel_levels['date'])
    fuel_levels = fuel_levels.sort_values('date')

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    window = 30
    rolling_mean = ridership['total_ridership'].rolling(window=window, center=True).mean()
    rolling_std = ridership['total_ridership'].rolling(window=window, center=True).std()

    ax1.plot(ridership['date'], rolling_mean, color='#2c3e50', linewidth=2, label=f'{window}-day Rolling Mean')
    ax1.fill_between(ridership['date'], rolling_mean - rolling_std, rolling_mean + rolling_std,
                    alpha=0.3, color='#3498db', label='±1 Std Dev')
    ax1.set_title('Ridership Rolling Mean with Variability', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total Ridership')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    mean_val = rolling_mean.mean()
    std_val = rolling_std.mean()
    changes = ridership.iloc[np.where(np.abs(rolling_mean - mean_val) > 2 * std_val)]['date'].tolist()
    for change_date in changes[:5]:
        ax1.axvline(x=change_date, color='#e74c3c', linestyle='--', alpha=0.7)

    ax2 = axes[1]
    rolling_fuel = fuel_levels.set_index('date')['ron95'].rolling(window=14, center=True).mean()
    ax2.plot(fuel_levels['date'], rolling_fuel, color='#e74c3c', linewidth=2, label='RON95 (14-day Rolling)')
    ax2.set_title('Fuel Price Rolling Average', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (MYR)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    fuel_mean = rolling_fuel.mean()
    fuel_std = rolling_fuel.std()
    fuel_changes = fuel_levels.iloc[np.where(np.abs(rolling_fuel - fuel_mean) > 1.5 * fuel_std)]['date'].tolist()
    for change_date in fuel_changes[:5]:
        ax2.axvline(x=change_date, color='#3498db', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'changepoint_detection.png', dpi=150, bbox_inches='tight')
    plt.close()

    changepoint_summary = pd.DataFrame({
        'ridership_changepoints': [str(c) for c in changes[:10]],
        'fuel_changepoints': [str(c) for c in fuel_changes[:10]]
    })
    changepoint_summary.to_csv(CSV_DIR / 'detected_changepoints.csv', index=False)


def run_temporal_eda():
    """Run all temporal EDA."""
    print("Loading data for temporal analysis...")
    fuel, rainfall, ridership, holidays = load_all_data()
    print(f"  Fuel data: {len(fuel)} rows")
    print(f"  Rainfall data: {len(rainfall)} rows")
    print(f"  Ridership data: {len(ridership)} rows")
    print(f"  Holidays: {len(holidays)} events")

    print("\nGenerating fuel price trends...")
    plot_fuel_price_trends(fuel)

    print("Generating ridership time series...")
    plot_ridership_timeseries(ridership)

    print("Generating rainfall seasonality...")
    plot_rainfall_seasonality(rainfall)

    print("Generating holiday calendar effects...")
    plot_holiday_calendar_effects(ridership, holidays)

    print("Generating change-point detection...")
    detect_changepoints(ridership, fuel)

    print(f"\nTemporal EDA complete. Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_temporal_eda()