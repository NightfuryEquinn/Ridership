"""
Demand Drivers Theme EDA
What causes ridership to go up or down? Fuel prices, rainfall events,
holiday calendars, and population distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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


def load_demand_driver_data():
    """Load all data for demand drivers analysis."""
    fuel = pd.read_csv(DATA_DIR / "fuelprice_cleaned.csv", parse_dates=['date'])
    rainfall = pd.read_csv(DATA_DIR / "rainfall_combined_final.csv", parse_dates=['date'])
    ridership = pd.read_csv(DATA_DIR / "ridership_headline_clean.csv", parse_dates=['date'])
    holidays = pd.read_csv(DATA_DIR / "school_public_holiday_clean.csv", parse_dates=['start_date', 'end_date'])
    population = pd.read_csv(DATA_DIR / "population_density_clean.csv")
    return fuel, rainfall, ridership, holidays, population


def prepare_merged_dataset(ridership, fuel, rainfall, holidays):
    """Merge all temporal datasets for correlation analysis."""
    ridership = ridership.copy()
    ridership['date'] = pd.to_datetime(ridership['date'])

    fuel_levels = fuel[fuel['series_type'] == 'level'].copy()
    fuel_levels['date'] = pd.to_datetime(fuel_levels['date'])

    rainfall_agg = rainfall.copy()
    rainfall_agg['date'] = pd.to_datetime(rainfall_agg['date'])
    rainfall_agg = rainfall_agg.groupby('date').agg({
        'rainfall_mm': 'mean',
        'anomaly_rf': 'mean'
    }).reset_index()

    holidays['start_date'] = pd.to_datetime(holidays['start_date'])
    holidays['end_date'] = pd.to_datetime(holidays['end_date'])
    holiday_dates = set()
    for _, row in holidays.iterrows():
        date_range = pd.date_range(start=row['start_date'], end=row['end_date'])
        holiday_dates.update(date_range)

    ridership['is_holiday'] = ridership['date'].isin(holiday_dates)
    ridership['dayofweek'] = ridership['date'].dt.dayofweek
    ridership['is_weekend'] = ridership['dayofweek'].isin([5, 6])
    ridership['month'] = ridership['date'].dt.month

    merged = ridership.merge(fuel_levels[['date', 'ron95', 'diesel']], on='date', how='left')
    merged = merged.merge(rainfall_agg, on='date', how='left')

    return merged


def plot_fuel_ridership_relationship(merged):
    """Analyze relationship between fuel prices and ridership."""
    fuel_data = merged.dropna(subset=['ron95', 'total_ridership'])

    if len(fuel_data) < 10:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.text(0.5, 0.5, 'Insufficient data for fuel-ridership analysis', ha='center', va='center', fontsize=14)
        plt.savefig(GRAPH_DIR / 'fuel_ridership_relationship.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    ax1.scatter(fuel_data['ron95'], fuel_data['total_ridership'], alpha=0.3, c='blue', s=10)
    z = np.polyfit(fuel_data['ron95'], fuel_data['total_ridership'], 1)
    p = np.poly1d(z)
    ax1.plot(fuel_data['ron95'].sort_values(), p(fuel_data['ron95'].sort_values()), "r--", linewidth=2, label=f'Trend (slope={z[0]:.0f})')
    ax1.set_title('Fuel Price vs Ridership', fontsize=12, fontweight='bold')
    ax1.set_xlabel('RON95 Price (MYR)')
    ax1.set_ylabel('Total Ridership')
    ax1.legend()

    corr, pval = stats.pearsonr(fuel_data['ron95'], fuel_data['total_ridership'])
    ax1.text(0.05, 0.95, f'r={corr:.3f}, p={pval:.3f}', transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = axes[0, 1]
    fuel_data_sorted = fuel_data.sort_values('ron95')
    try:
        fuel_bins = pd.qcut(fuel_data_sorted['ron95'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
    except ValueError:
        percentiles = fuel_data_sorted['ron95'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
        unique_bounds = []
        for p in percentiles:
            if not unique_bounds or unique_bounds[-1] != p:
                unique_bounds.append(p)
        if len(unique_bounds) < 3:
            fuel_bins = pd.cut(fuel_data_sorted['ron95'], bins=2, labels=['Low', 'High'])
        else:
            fuel_bins = pd.cut(fuel_data_sorted['ron95'], bins=unique_bounds, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'][:len(unique_bounds)-1])
    ridership_by_fuel = fuel_data_sorted.groupby(fuel_bins, observed=True)['total_ridership'].mean()
    n_bins = len(ridership_by_fuel)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, max(n_bins, 2)))
    ridership_by_fuel.plot(kind='bar', ax=ax2, color=colors[:n_bins])
    ax2.set_title('Average Ridership by Fuel Price Quintile', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Fuel Price Level')
    ax2.set_ylabel('Average Daily Ridership')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    ax3 = axes[1, 0]
    fuel_data['ron95_change'] = fuel_data['ron95'].pct_change() * 100
    fuel_data['ridership_change'] = fuel_data['total_ridership'].pct_change() * 100
    changes = fuel_data.dropna(subset=['ron95_change', 'ridership_change'])
    changes = changes[(np.abs(changes['ron95_change']) < 20) & (np.abs(changes['ridership_change']) < 30)]
    ax3.scatter(changes['ron95_change'], changes['ridership_change'], alpha=0.3, c='green', s=10)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Weekly Changes: Fuel Price vs Ridership', fontsize=12, fontweight='bold')
    ax3.set_xlabel('RON95 Price Change (%)')
    ax3.set_ylabel('Ridership Change (%)')

    ax4 = axes[1, 1]
    diesel_data = merged.dropna(subset=['diesel', 'total_ridership'])
    if len(diesel_data) > 10:
        ax4.scatter(diesel_data['diesel'], diesel_data['total_ridership'], alpha=0.3, c='orange', s=10)
        z_d = np.polyfit(diesel_data['diesel'], diesel_data['total_ridership'], 1)
        p_d = np.poly1d(z_d)
        ax4.plot(diesel_data['diesel'].sort_values(), p_d(diesel_data['diesel'].sort_values()), "r--", linewidth=2)
        ax4.set_title('Diesel Price vs Ridership', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Diesel Price (MYR)')
        ax4.set_ylabel('Total Ridership')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'fuel_ridership_relationship.png', dpi=150, bbox_inches='tight')
    plt.close()

    fuel_summary = pd.DataFrame({
        'metric': ['correlation_ron95', 'p_value_ron95', 'correlation_diesel', 'p_value_diesel'],
        'value': [corr, pval, np.nan, np.nan]
    })
    fuel_summary.to_csv(CSV_DIR / 'fuel_ridership_correlation.csv', index=False)


def plot_rainfall_ridership_impact(merged):
    """Analyze rainfall impact on ridership."""
    rain_data = merged.dropna(subset=['rainfall_mm', 'total_ridership'])

    if len(rain_data) < 10:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.text(0.5, 0.5, 'Insufficient data for rainfall-ridership analysis', ha='center', va='center', fontsize=14)
        plt.savefig(GRAPH_DIR / 'rainfall_ridership_impact.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    ax1.scatter(rain_data['rainfall_mm'], rain_data['total_ridership'], alpha=0.3, c='blue', s=10)
    ax1.set_title('Rainfall vs Ridership', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Rainfall (mm)')
    ax1.set_ylabel('Total Ridership')

    rain_corr, rain_pval = stats.pearsonr(rain_data['rainfall_mm'], rain_data['total_ridership'])
    ax1.text(0.05, 0.95, f'r={rain_corr:.3f}, p={rain_pval:.3f}', transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = axes[0, 1]
    rain_data['rain_category'] = pd.cut(rain_data['rainfall_mm'], bins=[0, 5, 15, 30, 100, 500],
                                         labels=['None (0-5)', 'Light (5-15)', 'Moderate (15-30)', 'Heavy (30-100)', 'Very Heavy (>100)'])
    ridership_by_rain = rain_data.groupby('rain_category')['total_ridership'].agg(['mean', 'std', 'count'])
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(ridership_by_rain)))
    ridership_by_rain['mean'].plot(kind='bar', ax=ax2, color=colors, yerr=ridership_by_rain['std'], capsize=3)
    ax2.set_title('Average Ridership by Rainfall Category', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Rainfall Category')
    ax2.set_ylabel('Average Daily Ridership')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    ax3 = axes[1, 0]
    rain_days = rain_data[rain_data['rainfall_mm'] > 10]
    no_rain_days = rain_data[rain_data['rainfall_mm'] <= 5]
    categories = ['No Rain\n(≤5mm)', 'Rain\n(>10mm)']
    means = [no_rain_days['total_ridership'].mean(), rain_days['total_ridership'].mean()]
    stds = [no_rain_days['total_ridership'].std(), rain_days['total_ridership'].std()]
    counts = [len(no_rain_days), len(rain_days)]
    bars = ax3.bar(categories, means, yerr=stds, capsize=5, color=['#2ecc71', '#3498db'])
    ax3.set_title('Ridership: Rain vs No Rain Days', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Daily Ridership')
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    ax4 = axes[1, 1]
    rain_data_sorted = rain_data.sort_values('rainfall_mm')
    ax4.scatter(rain_data_sorted['date'], rain_data_sorted['rainfall_mm'], c='blue', alpha=0.3, s=5, label='Rainfall')
    ax4.set_title('Rainfall Time Series', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Rainfall (mm)')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'rainfall_ridership_impact.png', dpi=150, bbox_inches='tight')
    plt.close()

    rainfall_summary = rain_data.groupby('rain_category')['total_ridership'].agg(['mean', 'std', 'count']).round(0)
    rainfall_summary.to_csv(CSV_DIR / 'ridership_by_rainfall_category.csv')


def plot_holiday_population_effects(merged, population):
    """Analyze holiday and population effects on ridership."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    day_type_stats = merged.groupby(['is_weekend', 'is_holiday'])['total_ridership'].mean().reset_index()
    day_type_stats['day_type'] = day_type_stats.apply(
        lambda x: 'Weekend+Holiday' if x['is_weekend'] and x['is_holiday']
        else 'Weekend' if x['is_weekend']
        else 'Holiday' if x['is_holiday']
        else 'Normal', axis=1
    )
    colors = {'Normal': '#3498db', 'Weekend': '#e74c3c', 'Holiday': '#2ecc71', 'Weekend+Holiday': '#f39c12'}
    bars = ax1.bar(day_type_stats['day_type'], day_type_stats['total_ridership'],
                   color=[colors.get(dt, 'gray') for dt in day_type_stats['day_type']])
    ax1.set_title('Average Ridership by Day Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Daily Ridership')
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)

    ax2 = axes[0, 1]
    monthly_ridership = merged.groupby('month')['total_ridership'].agg(['mean', 'std'])
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax2.bar(range(1, 13), monthly_ridership['mean'], yerr=monthly_ridership['std'], capsize=3, color='steelblue')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(months)
    ax2.set_title('Average Ridership by Month', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Average Daily Ridership')

    ax3 = axes[1, 0]
    pop_stats = population['density_per_km2'].describe()
    pop_bins = [0, 100, 500, 1000, 5000, 10000, 50000]
    pop_labels = ['Rural\n(<100)', 'Suburban\n(100-500)', 'Urban\n(500-1k)', 'Dense\n(1k-5k)', 'Urban\n(5k-10k)', 'Very Dense\n(>10k)']
    pop_data = pd.cut(population['density_per_km2'], bins=pop_bins, labels=pop_labels[:len(pop_bins)-1])
    pop_counts = pd.Series(pop_data).value_counts()
    ax3.bar(range(len(pop_counts)), pop_counts.values, color='coral')
    ax3.set_xticks(range(len(pop_counts)))
    ax3.set_xticklabels(pop_labels[:len(pop_counts)], rotation=45, ha='right')
    ax3.set_title('Population Distribution by Density Category', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Grid Cell Count')

    ax4 = axes[1, 1]
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_stats = merged.groupby('dayofweek')['total_ridership'].mean()
    colors_dow = ['#3498db' if i < 5 else '#e74c3c' for i in range(7)]
    ax4.bar(range(7), dow_stats.values, color=colors_dow)
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(dow_names)
    ax4.set_title('Average Ridership by Day of Week', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Day of Week')
    ax4.set_ylabel('Average Daily Ridership')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'holiday_population_effects.png', dpi=150, bbox_inches='tight')
    plt.close()

    holiday_effect = merged.groupby(['is_holiday', 'is_weekend'])['total_ridership'].agg(['mean', 'std', 'count']).round(0)
    holiday_effect.to_csv(CSV_DIR / 'ridership_holiday_weekend_effects.csv')


def plot_demand_driver_correlations(merged):
    """Correlation matrix of demand drivers."""
    cols_for_corr = ['total_ridership', 'ron95', 'diesel', 'rainfall_mm', 'is_holiday', 'is_weekend', 'month']
    available_cols = [c for c in cols_for_corr if c in merged.columns]

    if len(available_cols) < 3:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.text(0.5, 0.5, 'Insufficient data for correlation analysis', ha='center', va='center', fontsize=14)
        plt.savefig(GRAPH_DIR / 'demand_driver_correlations.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    corr_data = merged[available_cols].dropna()
    if len(corr_data) < 30:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.text(0.5, 0.5, 'Insufficient paired data for correlation', ha='center', va='center', fontsize=14)
        plt.savefig(GRAPH_DIR / 'demand_driver_correlations.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    corr_matrix = corr_data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)

    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)

    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Demand Driver Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'demand_driver_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()

    corr_matrix.to_csv(CSV_DIR / 'demand_driver_correlation_matrix.csv')


def run_demand_drivers_eda():
    """Run all demand drivers EDA."""
    print("Loading data for demand drivers analysis...")
    fuel, rainfall, ridership, holidays, population = load_demand_driver_data()
    print(f"  Fuel data: {len(fuel)} rows")
    print(f"  Rainfall data: {len(rainfall)} rows")
    print(f"  Ridership data: {len(ridership)} rows")
    print(f"  Holidays: {len(holidays)} events")
    print(f"  Population points: {len(population)}")

    print("\nPreparing merged dataset...")
    merged = prepare_merged_dataset(ridership, fuel, rainfall, holidays)

    print("Generating fuel-ridership relationship analysis...")
    plot_fuel_ridership_relationship(merged)

    print("Generating rainfall impact analysis...")
    plot_rainfall_ridership_impact(merged)

    print("Generating holiday and population effects...")
    plot_holiday_population_effects(merged, population)

    print("Generating demand driver correlations...")
    plot_demand_driver_correlations(merged)

    print(f"\nDemand Drivers EDA complete. Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_demand_drivers_eda()