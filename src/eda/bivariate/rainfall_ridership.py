#!/usr/bin/env python3
"""
Bivariate EDA: Rainfall vs Ridership
Objective: Weather sensitivity of transit usage

Analysis:
1. Correlation between rainfall (daily, 1-month, 3-month accumulation) and ridership
2. Extreme weather events: heavy rain days and ridership drops
3. Seasonal rainfall patterns and ridership seasonality interaction
4. Geographic sensitivity: does rainfall in different regions affect ridership differently?
5. Recovery analysis: how quickly does ridership recover after heavy rain days?
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.max_columns = None


def load_data():
    """Load ridership and rainfall data."""
    ridership = pd.read_csv(
        os.path.join(BASE_DIR, "../../../data/cleaned/ridership_headline_clean.csv"),
        parse_dates=['date']
    )
    ridership = ridership.set_index('date').sort_index()

    rainfall = pd.read_csv(
        os.path.join(BASE_DIR, "../../../data/cleaned/rainfall_combined_final.csv"),
        parse_dates=['date']
    )
    return ridership, rainfall


def prepare_national_rainfall(rainfall):
    """Aggregate rainfall to national daily level."""
    rf = rainfall.copy()
    rf['date'] = pd.to_datetime(rf['date'])

    national = rf.groupby('date').agg({
        'rainfall_mm': 'mean',
        'rainfall_avg_mm': 'mean',
        'acc_1mo_mm': 'mean',
        'acc_1mo_avg_mm': 'mean',
        'acc_3mo_mm': 'mean',
        'acc_3mo_avg_mm': 'mean',
        'anomaly_rf': 'mean',
        'anomaly_1mo': 'mean',
        'anomaly_3mo': 'mean'
    }).reset_index()
    national = national.set_index('date').sort_index()
    return national


def merge_rainfall_ridership(ridership, rainfall_national):
    """Merge rainfall and ridership at daily level."""
    merged = ridership.join(rainfall_national, how='inner')
    merged = merged.dropna(subset=['rainfall_mm', 'anomaly_1mo'])
    merged['day_of_week'] = merged.index.dayofweek
    merged['is_weekend'] = merged['day_of_week'] >= 5
    merged['rainfall_bin'] = pd.cut(
        merged['rainfall_mm'],
        bins=[0, 10, 30, 50, 100, 500],
        labels=['Light (0-10)', 'Moderate (10-30)', 'Heavy (30-50)', 'Very Heavy (50-100)', 'Extreme (>100)']
    )
    return merged


def rainfall_ridership_correlation(merged):
    """Correlation analysis between rainfall metrics and ridership."""
    print("  Running correlation analysis...")

    cols = ['total_ridership', 'rainfall_mm', 'rainfall_avg_mm',
            'acc_1mo_mm', 'acc_3mo_mm', 'anomaly_rf', 'anomaly_1mo']
    corr = merged[cols].corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(corr, annot=True, fmt='.3f', cmap='Blues', ax=axes[0],
                vmin=-1, vmax=1, square=True)
    axes[0].set_title('Correlation: Rainfall Metrics vs Ridership')

    rolling_corr = merged['total_ridership'].rolling(30).corr(merged['rainfall_mm'])
    axes[1].plot(rolling_corr.index, rolling_corr.values, color='steelblue', alpha=0.8)
    axes[1].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Rolling 30-day Correlation')
    axes[1].set_title('Rolling Correlation: Daily Rainfall vs Ridership')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rainfall_ridership_correlation.png'), dpi=150)
    plt.close()
    print("    Saved rainfall_ridership_correlation.png")

    print(f"    Correlation (total_ridership vs rainfall_mm): {corr.loc['total_ridership', 'rainfall_mm']:.4f}")
    print(f"    Correlation (total_ridership vs anomaly_1mo): {corr.loc['total_ridership', 'anomaly_1mo']:.4f}")


def rainfall_category_analysis(merged):
    """Analyze ridership by rainfall intensity category."""
    print("  Running rainfall category analysis...")

    category_stats = merged.groupby('rainfall_bin', observed=True)['total_ridership'].agg(
        ['mean', 'std', 'count']
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.boxplot(data=merged, x='rainfall_bin', y='total_ridership', ax=axes[0, 0])
    axes[0, 0].set_xlabel('Rainfall Category')
    axes[0, 0].set_ylabel('Total Ridership')
    axes[0, 0].set_title('Ridership Distribution by Rainfall Intensity')
    axes[0, 0].tick_params(axis='x', rotation=30)

    means = merged.groupby('rainfall_bin', observed=True)['total_ridership'].mean()
    stds = merged.groupby('rainfall_bin', observed=True)['total_ridership'].std()
    x_pos = range(len(means))
    axes[0, 1].bar(x_pos, means.values, yerr=stds.values, capsize=5,
                    color='steelblue', alpha=0.7)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(means.index, rotation=30)
    axes[0, 1].set_ylabel('Mean Ridership')
    axes[0, 1].set_title('Mean Ridership by Rainfall Category\n(with Std Dev)')

    pct_change = []
    for cat in merged['rainfall_bin'].dropna().unique():
        cat_data = merged[merged['rainfall_bin'] == cat]['total_ridership']
        baseline = merged['total_ridership'].mean()
        pct_change.append({
            'category': str(cat),
            'pct_diff': (cat_data.mean() - baseline) / baseline * 100
        })
    pct_df = pd.DataFrame(pct_change)
    colors = ['coral' if x < 0 else 'seagreen' for x in pct_df['pct_diff']]
    axes[1, 0].bar(pct_df['category'], pct_df['pct_diff'], color=colors, alpha=0.7)
    axes[1, 0].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[1, 0].set_ylabel('% Difference from Overall Mean')
    axes[1, 0].set_title('Ridership Deviation by Rainfall Category')
    axes[1, 0].tick_params(axis='x', rotation=30)

    rainy_days = merged[merged['rainfall_mm'] > 10]
    dry_days = merged[merged['rainfall_mm'] <= 10]
    if len(dry_days) > 0 and len(rainy_days) > 0:
        axes[1, 1].violinplot([dry_days['total_ridership'].dropna(), rainy_days['total_ridership'].dropna()],
                              positions=[1, 2], showmeans=True)
        axes[1, 1].set_xticks([1, 2])
        axes[1, 1].set_xticklabels(['Dry Days\n(<=10mm)', 'Rainy Days\n(>10mm)'])
        axes[1, 1].set_ylabel('Total Ridership')
        axes[1, 1].set_title('Ridership Distribution: Dry vs Rainy Days')
    else:
        axes[1, 1].text(0.5, 0.5,
                         f'All {len(rainy_days)} days have rainfall > 10mm\n(All days are rainy)',
                         ha='center', va='center')
        axes[1, 1].set_title('Ridership Distribution: Dry vs Rainy Days')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rainfall_ridership_category.png'), dpi=150)
    plt.close()
    print("    Saved rainfall_ridership_category.png")


def extreme_weather_events(merged):
    """Analyze ridership drops during extreme weather events."""
    print("  Running extreme weather events analysis...")

    threshold_95 = merged['rainfall_mm'].quantile(0.95)

    merged['extreme_rain'] = merged['rainfall_mm'] >= threshold_95

    extreme_days = merged[merged['extreme_rain']]
    normal_days = merged[~merged['extreme_rain']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    daily_ridership = merged['total_ridership']
    axes[0, 0].plot(daily_ridership.index, daily_ridership.values, alpha=0.5, color='steelblue')
    extreme_mask = merged['extreme_rain']
    axes[0, 0].scatter(merged.index[extreme_mask], merged['total_ridership'][extreme_mask],
                       color='coral', s=30, label=f'Extreme Rain (Top 5%, >{threshold_95:.0f}mm)', zorder=5)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Total Ridership')
    axes[0, 0].set_title('Ridership with Extreme Rain Days Highlighted')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)

    ridership_ratio = extreme_days['total_ridership'].mean() / normal_days['total_ridership'].mean() if len(normal_days) > 0 else 1
    data_for_bar = pd.DataFrame({
        'Category': ['Normal Days', 'Extreme Rain Days (Top 5%)'],
        'Mean Ridership': [normal_days['total_ridership'].mean(), extreme_days['total_ridership'].mean()],
        'Std': [normal_days['total_ridership'].std(), extreme_days['total_ridership'].std()]
    })
    axes[0, 1].bar(data_for_bar['Category'], data_for_bar['Mean Ridership'],
                   yerr=data_for_bar['Std'], capsize=5, color=['seagreen', 'coral'], alpha=0.8)
    axes[0, 1].set_ylabel('Mean Ridership')
    axes[0, 1].set_title(f'Residership Drop During Extreme Rain\n(Ratio: {ridership_ratio:.3f})')

    merged['rainfall_change'] = merged['rainfall_mm'].pct_change() * 100
    merged['ridership_change'] = merged['total_ridership'].pct_change() * 100
    valid = merged.dropna(subset=['rainfall_change', 'ridership_change'])
    valid = valid[(valid['rainfall_change'].abs() < 500)]

    axes[1, 0].scatter(valid['rainfall_change'], valid['ridership_change'],
                       alpha=0.3, s=10, color='steelblue')
    axes[1, 0].axhline(0, color='grey', linestyle='--', alpha=0.3)
    axes[1, 0].axvline(0, color='grey', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Rainfall Change (%)')
    axes[1, 0].set_ylabel('Ridership Change (%)')
    axes[1, 0].set_title('Ridership Change vs Rainfall Change')

    rain_percentiles = [10, 25, 50, 75, 90, 95, 99]
    pct_labels = ['P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99']

    recovery_data = []
    for pct, label in zip(rain_percentiles, pct_labels):
        threshold = merged['rainfall_mm'].quantile(pct / 100)
        high_rain = merged[merged['rainfall_mm'] >= threshold]
        if len(high_rain) >= 3:
            next_3_days = merged.loc[high_rain.index[0]:].iloc[:3]['total_ridership'].mean()
            before = high_rain.iloc[0:1]['total_ridership'].mean()
            recovery_data.append({'threshold': label, 'before': before, 'next_3d': next_3_days})

    if recovery_data:
        rec_df = pd.DataFrame(recovery_data)
        x = np.arange(len(rec_df))
        width = 0.35
        axes[1, 1].bar(x - width/2, rec_df['before'], width, label='Before', color='steelblue')
        axes[1, 1].bar(x + width/2, rec_df['next_3d'], width, label='Next 3 Days', color='coral')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(rec_df['threshold'])
        axes[1, 1].set_ylabel('Mean Ridership')
        axes[1, 1].set_title('Ridership Recovery After High Rainfall Days')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rainfall_ridership_extreme_events.png'), dpi=150)
    plt.close()
    print("    Saved rainfall_ridership_extreme_events.png")


def weekday_weekend_rainfall_interaction(merged):
    """Analyze rainfall impact separately for weekdays vs weekends."""
    print("  Running weekday/weekend interaction analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    weekday_data = merged[~merged['is_weekend']]
    weekend_data = merged[merged['is_weekend']]

    sns.boxplot(data=weekday_data, x='rainfall_bin', y='total_ridership',
                ax=axes[0, 0], order=merged['rainfall_bin'].cat.categories)
    axes[0, 0].set_title('Weekday Ridership by Rainfall')
    axes[0, 0].tick_params(axis='x', rotation=30)

    sns.boxplot(data=weekend_data, x='rainfall_bin', y='total_ridership',
                ax=axes[0, 1], order=merged['rainfall_bin'].cat.categories)
    axes[0, 1].set_title('Weekend Ridership by Rainfall')
    axes[0, 1].tick_params(axis='x', rotation=30)

    for is_weekend, label, color in [(False, 'Weekday', 'steelblue'), (True, 'Weekend', 'coral')]:
        subset = merged[merged['is_weekend'] == is_weekend]
        bins = pd.cut(subset['rainfall_mm'], bins=10)
        grouped = subset.groupby(bins, observed=True)['total_ridership'].mean()
        group_idx = [i for i in range(len(grouped))]
        axes[1, 0].plot(group_idx, grouped.values, marker='o', label=label, color=color)

    axes[1, 0].set_xlabel('Rainfall Decile')
    axes[1, 0].set_ylabel('Mean Ridership')
    axes[1, 0].set_title('Ridership vs Rainfall Decile\n(Weekday vs Weekend)')
    axes[1, 0].legend()

    for is_weekend, label, color in [(False, 'Weekday', 'steelblue'), (True, 'Weekend', 'coral')]:
        subset = merged[merged['is_weekend'] == is_weekend]
        subset_clean = subset.dropna(subset=['rainfall_mm', 'total_ridership'])
        if len(subset_clean) > 10:
            slope, intercept, r, p, _ = stats.linregress(
                subset_clean['rainfall_mm'], subset_clean['total_ridership']
            )
            x_line = np.linspace(subset_clean['rainfall_mm'].min(), subset_clean['rainfall_mm'].max(), 100)
            axes[1, 1].scatter(subset_clean['rainfall_mm'], subset_clean['total_ridership'],
                               alpha=0.2, s=10, color=color)
            axes[1, 1].plot(x_line, slope * x_line + intercept, color=color, linewidth=2,
                             label=f'{label} (r={r:.3f}, p={p:.4f})')

    axes[1, 1].set_xlabel('Rainfall (mm)')
    axes[1, 1].set_ylabel('Total Ridership')
    axes[1, 1].set_title('Linear Regression: Rainfall vs Ridership\n(Weekday vs Weekend)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rainfall_ridership_weekday_weekend.png'), dpi=150)
    plt.close()
    print("    Saved rainfall_ridership_weekday_weekend.png")


def monthly_seasonal_rainfall_pattern(merged):
    """Analyze monthly rainfall vs ridership seasonality interaction."""
    print("  Running monthly/seasonal pattern analysis...")

    merged['month'] = merged.index.month
    monthly = merged.groupby('month').agg({
        'rainfall_mm': 'mean',
        'total_ridership': 'mean'
    }).reset_index()
    monthly['month_name'] = pd.to_datetime(monthly['month'], format='%m').dt.strftime('%b')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax2 = ax1.twinx()
    ax1.bar(monthly['month'], monthly['rainfall_mm'], color='steelblue', alpha=0.5, label='Avg Rainfall (mm)')
    ax2.plot(monthly['month'], monthly['total_ridership'], color='coral', marker='o', linewidth=2, label='Avg Ridership')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Avg Daily Rainfall (mm)', color='steelblue')
    ax2.set_ylabel('Avg Total Ridership', color='coral')
    ax1.set_title('Monthly: Rainfall vs Ridership Seasonality')
    ax1.set_xticks(monthly['month'])
    ax1.set_xticklabels(monthly['month_name'])

    for m in merged['month'].unique():
        month_data = merged[merged['month'] == m]
        slope, _, r, p, _ = stats.linregress(
            month_data['rainfall_mm'], month_data['total_ridership']
        )
        axes[1].scatter(m, r, s=100, color='mediumpurple', zorder=5)
        axes[1].annotate(f'p={p:.3f}', (m, r), textcoords='offset points',
                             xytext=(0, 5), fontsize=7, ha='center')

    axes[1].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Correlation (r)')
    axes[1].set_title('Rainfall-Ridership Correlation by Month\n(with p-value)')
    axes[1].set_xticks(monthly['month'])
    axes[1].set_xticklabels(monthly['month_name'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rainfall_ridership_seasonal.png'), dpi=150)
    plt.close()
    print("    Saved rainfall_ridership_seasonal.png")


def main():
    print("=" * 60)
    print("Bivariate EDA: Rainfall vs Ridership")
    print("Objective: Weather sensitivity of transit usage")
    print("=" * 60)

    print("\nLoading data...")
    ridership, rainfall = load_data()
    print(f"  Ridership: {ridership.shape[0]} days, {ridership.shape[1]-1} services")
    print(f"  Rainfall: {rainfall.shape[0]} records, adm levels: {rainfall['adm_level'].unique()}")

    print("\nPreparing national daily rainfall aggregation...")
    rainfall_national = prepare_national_rainfall(rainfall)
    print(f"  National daily: {rainfall_national.shape[0]} days")

    print("\nMerging rainfall and ridership...")
    merged = merge_rainfall_ridership(ridership, rainfall_national)
    print(f"  Merged dataset: {merged.shape[0]} records")

    print("\nRunning analysis...")
    rainfall_ridership_correlation(merged)
    rainfall_category_analysis(merged)
    extreme_weather_events(merged)
    weekday_weekend_rainfall_interaction(merged)
    monthly_seasonal_rainfall_pattern(merged)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()