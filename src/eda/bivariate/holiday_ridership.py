#!/usr/bin/env python3
"""
Bivariate EDA: Holiday vs Ridership
Objective: Holiday impact on ridership pattern

Analysis:
1. Overall holiday vs normal day ridership comparison
2. Pre-holiday / during-holiday / post-holiday ridership profiles
3. Academic break vs public holiday vs term-time patterns
4. Duration effect: short holidays (1-3 days) vs long breaks (>5 days)
5. Day-of-week interaction: weekday holiday vs weekend holiday effect
6. Recovery analysis: post-holiday ridership return to normal
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
    """Load ridership and holiday data."""
    ridership = pd.read_csv(
        os.path.join(BASE_DIR, "../../../data/cleaned/ridership_headline_clean.csv"),
        parse_dates=['date']
    )
    ridership = ridership.set_index('date').sort_index()

    holidays = pd.read_csv(
        os.path.join(BASE_DIR, "../../../data/cleaned/school_public_holiday_clean.csv"),
        parse_dates=['start_date', 'end_date']
    )
    return ridership, holidays


def build_holiday_calendar(holidays, ridership_dates):
    """Build a full holiday calendar with pre/during/post flags."""
    holiday_df = holidays.copy()

    all_holiday_dates = set()
    pre_holiday_dates = set()
    post_holiday_dates = set()

    for _, row in holiday_df.iterrows():
        start = pd.Timestamp(row['start_date'])
        end = pd.Timestamp(row['end_date'])
        holiday_type = row['type']
        event_name = row['event_name']

        date_range = pd.date_range(start=start, end=end, freq='D')
        all_holiday_dates.update(date_range)

        for d in date_range:
            if d - pd.Timedelta(days=1) >= ridership_dates.min():
                pre_holiday_dates.add(d - pd.Timedelta(days=1))
            if d + pd.Timedelta(days=1) <= ridership_dates.max():
                post_holiday_dates.add(d + pd.Timedelta(days=1))

    ridership_idx = ridership_dates

    result = pd.DataFrame(index=ridership_idx)
    result['is_holiday'] = result.index.isin(all_holiday_dates).astype(int)
    result['is_pre_holiday'] = result.index.isin(pre_holiday_dates).astype(int)
    result['is_post_holiday'] = result.index.isin(post_holiday_dates).astype(int)
    result['is_holiday_period'] = ((result['is_holiday'] == 1) | (result['is_pre_holiday'] == 1) | (result['is_post_holiday'] == 1)).astype(int)

    for _, row in holiday_df.iterrows():
        start = pd.Timestamp(row['start_date'])
        end = pd.Timestamp(row['end_date'])
        date_range = pd.date_range(start=start, end=end, freq='D')
        mask = result.index.isin(date_range)
        result.loc[mask, 'holiday_type'] = row['type']
        result.loc[mask, 'holiday_name'] = row['event_name']
        result.loc[mask, 'duration_days'] = row['duration_days']

    result['holiday_type'] = result['holiday_type'].fillna('Normal')
    result['holiday_name'] = result['holiday_name'].fillna('Normal')
    result['duration_days'] = result['duration_days'].fillna(0)

    result['day_of_week'] = result.index.dayofweek
    result['is_weekend'] = result['day_of_week'] >= 5

    return result


def merge_ridership_holiday(ridership, holiday_calendar):
    """Merge ridership with holiday calendar."""
    merged = ridership.join(holiday_calendar, how='inner')
    merged['ridership_change'] = merged['total_ridership'].pct_change() * 100

    merged['holiday_regime'] = 'Normal'
    merged.loc[merged['is_pre_holiday'] == 1, 'holiday_regime'] = 'Pre-Holiday'
    merged.loc[merged['is_holiday'] == 1, 'holiday_regime'] = 'Holiday'
    merged.loc[merged['is_post_holiday'] == 1, 'holiday_regime'] = 'Post-Holiday'

    merged['duration_category'] = 'Normal'
    merged.loc[merged['duration_days'] >= 10, 'duration_category'] = 'Long Break (>=10d)'
    merged.loc[(merged['duration_days'] > 3) & (merged['duration_days'] < 10), 'duration_category'] = 'Medium Break (4-9d)'
    merged.loc[(merged['duration_days'] > 0) & (merged['duration_days'] <= 3), 'duration_category'] = 'Short Holiday (<=3d)'

    return merged


def overall_holiday_impact(merged):
    """Overall comparison: holiday vs normal day ridership."""
    print("  Running overall holiday impact analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    holiday_means = merged.groupby('is_holiday')['total_ridership'].agg(['mean', 'std', 'count'])
    holiday_means.index = ['Normal Day', 'Holiday']
    axes[0, 0].bar(holiday_means.index, holiday_means['mean'],
                   yerr=holiday_means['std'], capsize=5,
                   color=['seagreen', 'coral'], alpha=0.8)
    axes[0, 0].set_ylabel('Mean Total Ridership')
    axes[0, 0].set_title('Average Ridership: Holiday vs Normal Days')
    for i, (mean, count) in enumerate(zip(holiday_means['mean'], holiday_means['count'])):
        axes[0, 0].text(i, mean + holiday_means['std'].iloc[i] + 5000,
                        f'n={int(count)}', ha='center', fontsize=9)

    regime_stats = merged.groupby('holiday_regime')['total_ridership'].agg(['mean', 'std'])
    regime_order = ['Normal', 'Pre-Holiday', 'Holiday', 'Post-Holiday']
    regime_stats = regime_stats.reindex(regime_order)
    colors = ['seagreen', 'orange', 'coral', 'steelblue']
    axes[0, 1].bar(regime_stats.index, regime_stats['mean'],
                   yerr=regime_stats['std'], capsize=5, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('Mean Total Ridership')
    axes[0, 1].set_title('Ridership: Normal vs Pre/Holiday/Post-Holiday')

    holiday_days = merged[merged['is_holiday'] == 1]
    normal_days = merged[merged['is_holiday'] == 0]

    for col in ['bus_rkl', 'bus_rpn', 'rail_lrt_ampang', 'rail_mrt_kajang',
                'rail_lrt_kj', 'rail_monorail', 'rail_mrt_pjy']:
        if col in merged.columns:
            h_mean = holiday_days[col].mean()
            n_mean = normal_days[col].mean()
            pct_diff = (h_mean - n_mean) / n_mean * 100 if n_mean > 0 else 0

    service_cols = [c for c in merged.columns if c.startswith('bus_') or c.startswith('rail_')]
    pct_diffs = []
    for col in service_cols:
        if col in merged.columns:
            h_mean = holiday_days[col].mean()
            n_mean = normal_days[col].mean()
            pct_diff = (h_mean - n_mean) / n_mean * 100 if n_mean > 0 else 0
            pct_diffs.append({'service': col, 'pct_diff': pct_diff})

    pct_df = pd.DataFrame(pct_diffs).sort_values('pct_diff')
    colors = ['coral' if x < 0 else 'seagreen' for x in pct_df['pct_diff']]
    axes[1, 0].barh(pct_df['service'], pct_df['pct_diff'], color=colors, alpha=0.8)
    axes[1, 0].axvline(0, color='grey', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('% Change (Holiday vs Normal)')
    axes[1, 0].set_title('Service-Level Ridership Change\nDuring Holidays')

    daily_normal = normal_days['total_ridership'].values
    daily_holiday = holiday_days['total_ridership'].values
    violin_data = [daily_normal, daily_holiday]
    axes[1, 1].violinplot(violin_data, positions=[1, 2], showmeans=True)
    axes[1, 1].set_xticks([1, 2])
    axes[1, 1].set_xticklabels(['Normal Days', 'Holiday Days'])
    axes[1, 1].set_ylabel('Total Ridership')
    axes[1, 1].set_title('Ridership Distribution:\nNormal vs Holiday')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_ridership_overall_impact.png'), dpi=150)
    plt.close()
    print("    Saved holiday_ridership_overall_impact.png")

    pct_change = (holiday_days['total_ridership'].mean() - normal_days['total_ridership'].mean()) / normal_days['total_ridership'].mean() * 100
    print(f"    Holiday vs Normal ridership change: {pct_change:.2f}%")


def holiday_type_analysis(merged):
    """Analyze ridership by holiday type (Academic vs Public vs Normal)."""
    print("  Running holiday type analysis...")

    type_order = ['Normal', 'Academic', 'Public']
    available_types = [t for t in type_order if t in merged['holiday_type'].values]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    type_stats = merged.groupby('holiday_type')['total_ridership'].agg(['mean', 'std', 'count'])
    type_stats = type_stats.reindex([t for t in type_order if t in type_stats.index])

    colors = ['seagreen', 'steelblue', 'coral'][:len(type_stats)]
    axes[0, 0].bar(type_stats.index, type_stats['mean'],
                   yerr=type_stats['std'], capsize=5, color=colors, alpha=0.8)
    axes[0, 0].set_ylabel('Mean Total Ridership')
    axes[0, 0].set_title('Ridership by Holiday Type')

    type_ridership = merged.groupby(['holiday_type', 'is_weekend'])['total_ridership'].mean().unstack()
    type_ridership = type_ridership.reindex([t for t in type_order if t in type_ridership.index])
    type_ridership.columns = ['Weekday', 'Weekend']
    type_ridership.plot(kind='bar', ax=axes[0, 1], color=['steelblue', 'coral'], alpha=0.8)
    axes[0, 1].set_ylabel('Mean Total Ridership')
    axes[0, 1].set_title('Holiday Type: Weekday vs Weekend Effect')
    axes[0, 1].tick_params(axis='x', rotation=0)

    type_pct = merged.groupby('holiday_type')['ridership_change'].mean()
    type_pct = type_pct.reindex([t for t in type_order if t in type_pct.index])
    colors = ['seagreen' if x >= 0 else 'coral' for x in type_pct.values]
    axes[1, 0].bar(type_pct.index, type_pct.values, color=colors, alpha=0.8)
    axes[1, 0].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[1, 0].set_ylabel('Mean Daily Ridership Change (%)')
    axes[1, 0].set_title('Mean Daily Ridership Change by Holiday Type')

    events = merged[merged['holiday_name'] != 'Normal'].groupby('holiday_name')['total_ridership'].agg(['mean', 'count'])
    events = events[events['count'] >= 2].sort_values('mean')
    if len(events) > 0:
        axes[1, 1].barh(events.index, events['mean'], color='mediumpurple', alpha=0.8)
        axes[1, 1].set_xlabel('Mean Ridership')
        axes[1, 1].set_title('Ridership by Individual Holiday Event\n(min 2 occurrences)')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient multi-occurrence events', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_ridership_type_analysis.png'), dpi=150)
    plt.close()
    print("    Saved holiday_ridership_type_analysis.png")


def duration_effect(merged):
    """Analyze the effect of holiday duration on ridership."""
    print("  Running duration effect analysis...")

    merged['duration_category'] = pd.Categorical(
        merged['duration_category'],
        categories=['Normal', 'Short Holiday (<=3d)', 'Medium Break (4-9d)', 'Long Break (>=10d)'],
        ordered=True
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    dur_order = ['Normal', 'Short Holiday (<=3d)', 'Medium Break (4-9d)', 'Long Break (>=10d)']
    dur_stats = merged.groupby('duration_category', observed=True)['total_ridership'].agg(['mean', 'std', 'count'])
    dur_stats = dur_stats.reindex([d for d in dur_order if d in dur_stats.index])

    axes[0, 0].bar(dur_stats.index, dur_stats['mean'],
                   yerr=dur_stats['std'], capsize=5, color='steelblue', alpha=0.8)
    axes[0, 0].set_ylabel('Mean Total Ridership')
    axes[0, 0].set_title('Ridership by Holiday Duration')
    axes[0, 0].tick_params(axis='x', rotation=20)

    pct_from_normal = []
    normal_mean = dur_stats.loc['Normal', 'mean'] if 'Normal' in dur_stats.index else merged['total_ridership'].mean()
    for cat in dur_stats.index:
        pct_from_normal.append((dur_stats.loc[cat, 'mean'] - normal_mean) / normal_mean * 100)
    axes[0, 1].bar(dur_stats.index, pct_from_normal,
                   color=['seagreen' if x >= 0 else 'coral' for x in pct_from_normal], alpha=0.8)
    axes[0, 1].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylabel('% Difference from Normal')
    axes[0, 1].set_title('Ridership Deviation from Normal by Duration')
    axes[0, 1].tick_params(axis='x', rotation=20)

    merged['duration_bin'] = pd.cut(merged['duration_days'], bins=[-0.001, 0, 1, 3, 7, 14, float('inf')],
                                    labels=['Normal', '1 day', '2-3 days', '4-7 days', '8-14 days', '>14 days'])

    for duration, group in merged.groupby('duration_bin', observed=True):
        if duration != 'Normal' and len(group) > 0:
            axes[1, 0].hist(group['total_ridership'], bins=20, alpha=0.5, label=str(duration))
    axes[1, 0].set_xlabel('Total Ridership')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Ridership Distribution by Duration')
    axes[1, 0].legend()

    for duration, group in merged.groupby('duration_bin', observed=True):
        if len(group) > 5:
            group_clean = group.dropna(subset=['day_of_week', 'total_ridership'])
            if len(group_clean) > 5:
                group_mean = group_clean.groupby('day_of_week')['total_ridership'].mean()
                axes[1, 1].plot(group_mean.index, group_mean.values, marker='o',
                                 linewidth=2, label=str(duration))

    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Mean Ridership')
    axes[1, 1].set_title('Day-of-Week Profile by Holiday Duration')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_ridership_duration_effect.png'), dpi=150)
    plt.close()
    print("    Saved holiday_ridership_duration_effect.png")


def pre_post_holiday_profile(merged):
    """Profile ridership in pre-holiday, during-holiday, and post-holiday periods."""
    print("  Running pre/post-holiday profile analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    regime_order = ['Normal', 'Pre-Holiday', 'Holiday', 'Post-Holiday']
    regime_stats = merged.groupby('holiday_regime')['total_ridership'].agg(['mean', 'std', 'count'])
    regime_stats = regime_stats.reindex([r for r in regime_order if r in regime_stats.index])
    colors = ['seagreen', 'orange', 'coral', 'steelblue'][:len(regime_stats)]

    axes[0, 0].bar(regime_stats.index, regime_stats['mean'],
                   yerr=regime_stats['std'], capsize=5, color=colors, alpha=0.8)
    axes[0, 0].set_ylabel('Mean Total Ridership')
    axes[0, 0].set_title('Ridership: Pre-Holiday / During / Post-Holiday')

    holiday_merged = merged[merged['is_holiday_period'] == 1].copy()
    for regime, color in [('Pre-Holiday', 'orange'), ('Holiday', 'coral'), ('Post-Holiday', 'steelblue')]:
        subset = holiday_merged[holiday_merged['holiday_regime'] == regime]
        if len(subset) > 0:
            dow_mean = subset.groupby('day_of_week')['total_ridership'].mean()
            axes[0, 1].plot(dow_mean.index, dow_mean.values, marker='o',
                             linewidth=2, label=regime, color=color)

    normal_dow = merged[merged['holiday_regime'] == 'Normal'].groupby('day_of_week')['total_ridership'].mean()
    axes[0, 1].plot(normal_dow.index, normal_dow.values, marker='s',
                     linewidth=2, linestyle='--', color='seagreen', label='Normal')
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Mean Ridership')
    axes[0, 1].set_title('Day-of-Week Profile:\nPre-Holiday / Holiday / Post-Holiday vs Normal')
    axes[0, 1].legend()

    for is_weekend, label in [(False, 'Weekday'), (True, 'Weekend')]:
        subset = merged[merged['is_weekend'] == is_weekend]
        regime_stats = subset.groupby('holiday_regime')['total_ridership'].mean()
        regime_stats = regime_stats.reindex([r for r in regime_order if r in regime_stats.index])
        axes[1, 0].plot(regime_stats.index, regime_stats.values, marker='o',
                          linewidth=2, label=label)

    axes[1, 0].set_ylabel('Mean Total Ridership')
    axes[1, 0].set_title('Weekday vs Weekend Response\nto Holiday Periods')
    axes[1, 0].legend()

    change_stats = merged.groupby('holiday_regime')['ridership_change'].agg(['mean', 'std'])
    change_stats = change_stats.reindex([r for r in regime_order if r in change_stats.index])
    colors = ['seagreen' if x >= 0 else 'coral' for x in change_stats['mean']]
    axes[1, 1].bar(change_stats.index, change_stats['mean'],
                   yerr=change_stats['std'], capsize=5, color=colors, alpha=0.8)
    axes[1, 1].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[1, 1].set_ylabel('Mean Daily Ridership Change (%)')
    axes[1, 1].set_title('Daily Ridership Change by Holiday Regime')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_ridership_pre_post_profile.png'), dpi=150)
    plt.close()
    print("    Saved holiday_ridership_pre_post_profile.png")


def weekday_holiday_vs_weekend_holiday(merged):
    """Analyze impact when holiday falls on weekday vs weekend."""
    print("  Running weekday-holiday vs weekend-holiday analysis...")

    merged['holiday_on_weekday'] = ((merged['is_holiday'] == 1) & (merged['is_weekend'] == False)).astype(int)
    merged['holiday_on_weekend'] = ((merged['is_holiday'] == 1) & (merged['is_weekend'] == True)).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    categories = ['Normal', 'Weekend Holiday', 'Weekday Holiday']
    holiday_weekday = merged[merged['holiday_on_weekday'] == 1]['total_ridership'].mean()
    holiday_weekend = merged[merged['holiday_on_weekend'] == 1]['total_ridership'].mean()
    normal_mean = merged[(merged['holiday_on_weekday'] == 0) & (merged['holiday_on_weekend'] == 0)]['total_ridership'].mean()

    values = [normal_mean, holiday_weekend, holiday_weekday]
    counts = [
        len(merged[(merged['holiday_on_weekday'] == 0) & (merged['holiday_on_weekend'] == 0)]),
        merged['holiday_on_weekend'].sum(),
        merged['holiday_on_weekday'].sum()
    ]
    colors = ['seagreen', 'coral', 'steelblue']

    axes[0, 0].bar(categories, values, color=colors, alpha=0.8)
    axes[0, 0].set_ylabel('Mean Total Ridership')
    axes[0, 0].set_title('Normal Day vs Weekend Holiday vs Weekday Holiday')
    for i, (v, c) in enumerate(zip(values, counts)):
        axes[0, 0].text(i, v + 5000, f'n={int(c)}', ha='center', fontsize=9)

    pct_from_normal = [
        0,
        (holiday_weekend - normal_mean) / normal_mean * 100,
        (holiday_weekday - normal_mean) / normal_mean * 100
    ]
    axes[0, 1].bar(categories, pct_from_normal,
                   color=['grey', 'coral', 'steelblue'], alpha=0.8)
    axes[0, 1].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylabel('% Change from Normal')
    axes[0, 1].set_title('Ridership Change:\nWeekend Holiday vs Weekday Holiday')

    service_cols = [c for c in merged.columns if c.startswith('bus_') or c.startswith('rail_')]
    service_impact = []
    for col in service_cols:
        if col in merged.columns:
            hw = merged[merged['holiday_on_weekday'] == 1][col].mean()
            hwe = merged[merged['holiday_on_weekend'] == 1][col].mean()
            n = merged[(merged['holiday_on_weekday'] == 0) & (merged['holiday_on_weekend'] == 0)][col].mean()
            if n > 0:
                service_impact.append({
                    'service': col,
                    'weekday_holiday_pct': (hw - n) / n * 100,
                    'weekend_holiday_pct': (hwe - n) / n * 100
                })

    impact_df = pd.DataFrame(service_impact).sort_values('weekday_holiday_pct')

    x = np.arange(len(impact_df))
    width = 0.35
    axes[1, 0].barh(x - width/2, impact_df['weekday_holiday_pct'], width,
                    label='Weekday Holiday', color='steelblue', alpha=0.8)
    axes[1, 0].barh(x + width/2, impact_df['weekend_holiday_pct'], width,
                    label='Weekend Holiday', color='coral', alpha=0.8)
    axes[1, 0].axvline(0, color='grey', linestyle='--', alpha=0.5)
    axes[1, 0].set_yticks(x)
    axes[1, 0].set_yticklabels(impact_df['service'])
    axes[1, 0].set_xlabel('% Change from Normal')
    axes[1, 0].set_title('Service-Level Impact:\nWeekday Holiday vs Weekend Holiday')
    axes[1, 0].legend()

    combined_holiday = merged[(merged['holiday_on_weekday'] == 1) | (merged['holiday_on_weekend'] == 1)]
    dow_holiday = combined_holiday.groupby('day_of_week')['total_ridership'].mean()
    dow_normal = merged[(merged['holiday_on_weekday'] == 0) & (merged['holiday_on_weekend'] == 0)].groupby('day_of_week')['total_ridership'].mean()

    axes[1, 1].plot(dow_holiday.index, dow_holiday.values, marker='o',
                     linewidth=2, color='coral', label='Holiday')
    axes[1, 1].plot(dow_normal.index, dow_normal.values, marker='s',
                     linewidth=2, color='seagreen', label='Normal')
    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Mean Ridership')
    axes[1, 1].set_title('Day-of-Week Profile:\nHoliday Days vs Normal Days')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_ridership_weekday_vs_weekend.png'), dpi=150)
    plt.close()
    print("    Saved holiday_ridership_weekday_vs_weekend.png")


def main():
    print("=" * 60)
    print("Bivariate EDA: Holiday vs Ridership")
    print("Objective: Holiday impact on ridership pattern")
    print("=" * 60)

    print("\nLoading data...")
    ridership, holidays = load_data()
    ridership_dates = ridership.index
    print(f"  Ridership: {ridership.shape[0]} days, {ridership.shape[1]-1} services")
    print(f"  Holidays: {holidays.shape[0]} events")
    print(f"    Types: {holidays['type'].value_counts().to_dict()}")
    print(f"    Year range: {holidays['year'].min()} - {holidays['year'].max()}")

    print("\nBuilding holiday calendar...")
    holiday_calendar = build_holiday_calendar(holidays, ridership_dates)
    n_holiday_days = holiday_calendar['is_holiday'].sum()
    n_pre_days = holiday_calendar['is_pre_holiday'].sum()
    n_post_days = holiday_calendar['is_post_holiday'].sum()
    print(f"  Holiday days: {n_holiday_days}, Pre-holiday: {n_pre_days}, Post-holiday: {n_post_days}")

    print("\nMerging ridership with holiday calendar...")
    merged = merge_ridership_holiday(ridership, holiday_calendar)
    print(f"  Merged dataset: {merged.shape[0]} records")
    print(f"  Holiday type distribution:\n{merged['holiday_type'].value_counts().to_dict()}")

    print("\nRunning analysis...")
    overall_holiday_impact(merged)
    holiday_type_analysis(merged)
    duration_effect(merged)
    pre_post_holiday_profile(merged)
    weekday_holiday_vs_weekend_holiday(merged)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()