#!/usr/bin/env python3
"""
Bivariate EDA: Fuel Price vs Ridership
Objective: Fuel price elasticity of transit demand

Analysis:
1. Correlation between fuel prices (RON95, RON97, Diesel) and total ridership
2. Price elasticity estimation (ridership % change / fuel price % change)
3. Lagged effects (does ridership respond to fuel price changes with a delay?)
4. Mode substitution analysis (do rail services gain more when fuel prices rise?)
5. Threshold effects (is there a fuel price level that triggers ridership shift?)
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
    """Load and merge ridership and fuel price data."""
    ridership = pd.read_csv(
        os.path.join(BASE_DIR, "../../../data/cleaned/ridership_headline_clean.csv"),
        parse_dates=['date']
    )
    ridership = ridership.set_index('date').sort_index()

    fuel = pd.read_csv(
        os.path.join(BASE_DIR, "../../../data/cleaned/fuelprice_cleaned.csv"),
        parse_dates=['date']
    )
    return ridership, fuel


def merge_fuel_ridership(ridership, fuel):
    """Merge fuel and ridership data at appropriate granularity."""
    fuel_levels = fuel[fuel['series_type'] == 'level'].copy()
    fuel_levels = fuel_levels.set_index('date').sort_index()

    merged = ridership.join(fuel_levels, how='inner')
    fuel_cols = ['ron95', 'ron97', 'diesel', 'diesel_eastmsia']
    service_cols = [c for c in ridership.columns if c != 'total_ridership']
    merged = merged[service_cols + ['total_ridership'] + fuel_cols].copy()
    merged = merged.dropna()

    merged['ridership_change'] = merged['total_ridership'].pct_change() * 100
    merged['ron95_change'] = merged['ron95'].pct_change() * 100
    merged['diesel_change'] = merged['diesel'].pct_change() * 100

    return merged


def elasticity_analysis(merged):
    """Estimate price elasticity of transit demand."""
    print("  Running elasticity analysis...")

    valid = merged.dropna(subset=['ridership_change', 'ron95_change'])
    valid = valid[(valid['ron95_change'].abs() > 0.1) & (valid['ridership_change'].abs() > 0.1)]

    if len(valid) > 5:
        slope_ron95, intercept, r_ron95, p_ron95, _ = stats.linregress(
            valid['ron95_change'], valid['ridership_change']
        )

        diesel_valid = merged.dropna(subset=['ridership_change', 'diesel_change'])
        diesel_valid = diesel_valid[(diesel_valid['diesel_change'].abs() > 0.1)]
        if len(diesel_valid) > 5:
            slope_diesel, _, r_diesel, p_diesel, _ = stats.linregress(
                diesel_valid['diesel_change'], diesel_valid['ridership_change']
            )
        else:
            slope_diesel, r_diesel, p_diesel = np.nan, np.nan, np.nan

        print(f"    RON95 elasticity: {slope_ron95:.4f} (r={r_ron95:.3f}, p={p_ron95:.4f})")
        print(f"    Diesel elasticity: {slope_diesel:.4f} (r={r_diesel:.3f}, p={p_diesel:.4f})")
    else:
        slope_ron95, r_ron95, slope_diesel, r_diesel = np.nan, np.nan, np.nan, np.nan

    return slope_ron95, r_ron95, slope_diesel, r_diesel


def lagged_effects(merged):
    """Analyze lagged responses of ridership to fuel price changes."""
    print("  Running lagged effects analysis...")

    lags = range(0, 8)
    ron95_corrs = []
    diesel_corrs = []

    for lag in lags:
        shifted = merged['ron95_change'].shift(-lag)
        corr = merged['ridership_change'].corr(shifted)
        ron95_corrs.append(corr)

        diesel_shifted = merged['diesel_change'].shift(-lag)
        diesel_corr = merged['ridership_change'].corr(diesel_shifted)
        diesel_corrs.append(diesel_corr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(lags), ron95_corrs, marker='o', color='steelblue')
    axes[0].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Lag (days)')
    axes[0].set_ylabel('Correlation')
    axes[0].set_title('RON95 Price Change → Ridership Change\n(Lagged Correlation)')

    axes[1].plot(list(lags), diesel_corrs, marker='o', color='coral')
    axes[1].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Lag (days)')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Diesel Price Change → Ridership Change\n(Lagged Correlation)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fuel_ridership_lagged_effects.png'), dpi=150)
    plt.close()
    print("    Saved fuel_ridership_lagged_effects.png")

    best_lag_ron95 = list(lags)[np.nanargmax(np.abs(ron95_corrs))]
    best_lag_diesel = list(lags)[np.nanargmax(np.abs(diesel_corrs))]
    print(f"    Best lag for RON95: {best_lag_ron95} days (corr={ron95_corrs[best_lag_ron95]:.3f})")
    print(f"    Best lag for Diesel: {best_lag_diesel} days (corr={diesel_corrs[best_lag_diesel]:.3f})")


def correlation_heatmap(merged):
    """Correlation heatmap between fuel prices and ridership."""
    print("  Running correlation heatmap...")

    cols = ['total_ridership', 'ron95', 'ron97', 'diesel',
            'ridership_change', 'ron95_change', 'diesel_change']
    corr = merged[cols].corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=axes[0], vmin=-1, vmax=1, mask=mask, square=True)
    axes[0].set_title('Correlation: Fuel Prices vs Ridership\n(diesel_eastmsia excluded — constant at RM2.15)')

    rolling_corr = merged['total_ridership'].rolling(30).corr(merged['ron95'])
    axes[1].plot(rolling_corr.index, rolling_corr.values, color='steelblue', alpha=0.8)
    axes[1].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Rolling 30-day Correlation')
    axes[1].set_title('Rolling 30-Day Correlation:\nRON95 vs Total Ridership')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fuel_ridership_correlation.png'), dpi=150)
    plt.close()
    print("    Saved fuel_ridership_correlation.png")


def mode_substitution(merged):
    """Analyze if rail services gain more when fuel prices rise."""
    print("  Running mode substitution analysis...")

    service_cols = [c for c in merged.columns if c not in
                    ['total_ridership', 'ron95', 'ron97', 'diesel', 'diesel_eastmsia',
                     'ridership_change', 'ron95_change', 'diesel_change', 'price_regime',
                     'diesel_regime']]

    merged['price_regime'] = np.where(merged['ron95_change'] > 0, 'Price_Up', 'Price_Down')

    mode_impact = {}
    for col in service_cols:
        if col in merged.columns:
            up = merged[merged['price_regime'] == 'Price_Up'][col].mean()
            down = merged[merged['price_regime'] == 'Price_Down'][col].mean()
            if down > 0:
                mode_impact[col] = (up - down) / down * 100
            elif up > 0:
                mode_impact[col] = 0.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    mode_df = pd.DataFrame.from_dict(mode_impact, orient='index', columns=['pct_change'])
    mode_df = mode_df.sort_values('pct_change')
    if len(mode_df) > 0:
        axes[0, 0].barh(mode_df.index, mode_df['pct_change'],
                        color=['coral' if x < 0 else 'seagreen' for x in mode_df['pct_change']])
        axes[0, 0].axvline(0, color='grey', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('% Change in Ridership (Price Up vs Down)')
    axes[0, 0].set_title('Mode Substitution Effect:\nRidership When Fuel Price Increases vs Decreases')

    for col in service_cols[:5]:
        if col in merged.columns:
            axes[0, 1].scatter(merged['ron95'], merged[col], alpha=0.3, label=col, s=10)
    axes[0, 1].set_xlabel('RON95 Price (RM)')
    axes[0, 1].set_ylabel('Ridership')
    axes[0, 1].set_title('Fuel Price Level vs Ridership by Mode (First 5)')
    axes[0, 1].legend(fontsize=7)

    if len(service_cols) > 5:
        for col in service_cols[5:10]:
            if col in merged.columns:
                axes[1, 0].scatter(merged['ron95'], merged[col], alpha=0.3, label=col, s=10)
    axes[1, 0].set_xlabel('RON95 Price (RM)')
    axes[1, 0].set_ylabel('Ridership')
    axes[1, 0].set_title('Fuel Price Level vs Ridership by Mode (Next 5)')
    axes[1, 0].legend(fontsize=7)

    merged['diesel_regime'] = np.where(merged['diesel_change'] > 0, 'Diesel_Up', 'Diesel_Down')
    regime_comparison = merged.groupby('diesel_regime')['total_ridership'].mean()
    axes[1, 1].bar(regime_comparison.index, regime_comparison.values, color=['seagreen', 'coral'])
    axes[1, 1].set_ylabel('Average Total Ridership')
    axes[1, 1].set_title('Average Ridership by Diesel Price Change Direction')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fuel_ridership_mode_substitution.png'), dpi=150)
    plt.close()
    print("    Saved fuel_ridership_mode_substitution.png")


def threshold_analysis(merged):
    """Detect thresholds in fuel price that trigger ridership shifts."""
    print("  Running threshold analysis...")

    price_grid = np.linspace(merged['ron95'].quantile(0.1), merged['ron95'].quantile(0.9), 20)

    thresholds = []
    for p in price_grid:
        above = merged[merged['ron95'] >= p]['ridership_change'].mean()
        below = merged[merged['ron95'] < p]['ridership_change'].mean()
        thresholds.append({'price_threshold': p, 'above_mean': above, 'below_mean': below})

    th_df = pd.DataFrame(thresholds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(th_df['price_threshold'], th_df['above_mean'], label='Above threshold', color='coral')
    axes[0].plot(th_df['price_threshold'], th_df['below_mean'], label='Below threshold', color='steelblue')
    axes[0].set_xlabel('RON95 Price Threshold (RM)')
    axes[0].set_ylabel('Mean Ridership Change (%)')
    axes[0].set_title('Ridership Response Across Price Thresholds')
    axes[0].legend()

    diff = th_df['above_mean'] - th_df['below_mean']
    axes[1].bar(th_df['price_threshold'], diff, color='mediumpurple', alpha=0.7)
    axes[1].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('RON95 Price Threshold (RM)')
    axes[1].set_ylabel('Difference in Ridership Change (%)')
    axes[1].set_title('Price Threshold Impact:\n(Above - Below Threshold Ridership Change)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fuel_ridership_threshold.png'), dpi=150)
    plt.close()
    print("    Saved fuel_ridership_threshold.png")


def scatter_regression_plots(merged):
    """Scatter plots with regression lines for fuel vs ridership."""
    print("  Running scatter regression analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].scatter(merged['ron95'], merged['total_ridership'], alpha=0.4, s=10, color='steelblue')
    z = np.polyfit(merged['ron95'].dropna(), merged['total_ridership'].loc[merged['ron95'].notna()], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged['ron95'].min(), merged['ron95'].max(), 100)
    axes[0, 0].plot(x_line, p(x_line), color='coral', linewidth=2)
    axes[0, 0].set_xlabel('RON95 Price (RM)')
    axes[0, 0].set_ylabel('Total Ridership')
    axes[0, 0].set_title('RON95 vs Total Ridership')

    axes[0, 1].scatter(merged['diesel'], merged['total_ridership'], alpha=0.4, s=10, color='seagreen')
    valid = merged.dropna(subset=['diesel', 'total_ridership'])
    z = np.polyfit(valid['diesel'], valid['total_ridership'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['diesel'].min(), valid['diesel'].max(), 100)
    axes[0, 1].plot(x_line, p(x_line), color='coral', linewidth=2)
    axes[0, 1].set_xlabel('Diesel Price (RM)')
    axes[0, 1].set_ylabel('Total Ridership')
    axes[0, 1].set_title('Diesel vs Total Ridership')

    valid_change = merged.dropna(subset=['ron95_change', 'ridership_change'])
    axes[1, 0].scatter(valid_change['ron95_change'], valid_change['ridership_change'],
                       alpha=0.4, s=10, color='mediumpurple')
    axes[1, 0].set_xlabel('RON95 Price Change (%)')
    axes[1, 0].set_ylabel('Ridership Change (%)')
    axes[1, 0].set_title('RON95 Price Change vs Ridership Change')
    axes[1, 0].axhline(0, color='grey', linestyle='--', alpha=0.3)
    axes[1, 0].axvline(0, color='grey', linestyle='--', alpha=0.3)

    valid_diesel = merged.dropna(subset=['diesel_change', 'ridership_change'])
    axes[1, 1].scatter(valid_diesel['diesel_change'], valid_diesel['ridership_change'],
                       alpha=0.4, s=10, color='coral')
    axes[1, 1].set_xlabel('Diesel Price Change (%)')
    axes[1, 1].set_ylabel('Ridership Change (%)')
    axes[1, 1].set_title('Diesel Price Change vs Ridership Change')
    axes[1, 1].axhline(0, color='grey', linestyle='--', alpha=0.3)
    axes[1, 1].axvline(0, color='grey', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fuel_ridership_scatter_regression.png'), dpi=150)
    plt.close()
    print("    Saved fuel_ridership_scatter_regression.png")


def main():
    print("=" * 60)
    print("Bivariate EDA: Fuel Price vs Ridership")
    print("Objective: Fuel price elasticity of transit demand")
    print("=" * 60)

    print("\nLoading data...")
    ridership, fuel = load_data()
    print(f"  Ridership: {ridership.shape[0]} days, {ridership.shape[1]-1} services")
    print(f"  Fuel price: {fuel.shape[0]} records, types: {fuel['series_type'].unique()}")

    print("\nMerging fuel and ridership data...")
    merged = merge_fuel_ridership(ridership, fuel)
    print(f"  Merged dataset: {merged.shape[0]} records")

    print("\nRunning analysis...")
    e_ron95, r_ron95, e_diesel, r_diesel = elasticity_analysis(merged)
    lagged_effects(merged)
    correlation_heatmap(merged)
    mode_substitution(merged)
    threshold_analysis(merged)
    scatter_regression_plots(merged)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()