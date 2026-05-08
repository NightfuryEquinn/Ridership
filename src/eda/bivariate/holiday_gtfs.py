#!/usr/bin/env python3
"""
Bivariate EDA: Holiday vs GTFS Schedule Alignment
Objective: Service schedule alignment with demand cycles

Analysis:
1. GTFS service pattern: weekday vs weekend service availability
2. Holiday vs normal day ridership comparison
3. Service schedule adequacy during holiday periods (by day-of-week)
4. Duration effect: short holidays vs long breaks on service alignment
5. Temporal gap analysis: service frequency vs ridership demand by period
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

DATA_DIR = os.path.join(BASE_DIR, "../../../data/cleaned")
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.max_columns = None


def load_holidays():
    print("  Loading holiday data...")
    holidays = pd.read_csv(os.path.join(DATA_DIR, "school_public_holiday_clean.csv"),
                           parse_dates=['start_date', 'end_date'])
    print(f"    Loaded {len(holidays)} holiday records")
    return holidays


def load_ridership():
    print("  Loading ridership data...")
    ridership = pd.read_csv(os.path.join(DATA_DIR, "ridership_headline_clean.csv"),
                            parse_dates=['date'])
    ridership = ridership.set_index('date').sort_index()
    print(f"    Loaded {len(ridership):,} ridership records")
    return ridership


def load_all_gtfs():
    print("  Loading GTFS data...")
    gtfs_dirs = ['gtfs_ktmb', 'gtfs_rapid_bus_kl', 'gtfs_rapid_rail_kl']
    all_trips = []
    all_calendar = []
    all_stop_times = []

    for dir_name in gtfs_dirs:
        try:
            trips_file = os.path.join(DATA_DIR, dir_name, "trips.txt")
            cal_file = os.path.join(DATA_DIR, dir_name, "calendar.txt")
            st_file = os.path.join(DATA_DIR, dir_name, "stop_times.txt")

            trips = pd.read_csv(trips_file)
            trips['source'] = dir_name
            all_trips.append(trips)

            cal = pd.read_csv(cal_file)
            cal['source'] = dir_name
            all_calendar.append(cal)

            st = pd.read_csv(st_file)
            st['source'] = dir_name
            all_stop_times.append(st)
        except Exception as e:
            print(f"    Error loading {dir_name}: {e}")

    trips_df = pd.concat(all_trips, ignore_index=True)
    cal_df = pd.concat(all_calendar, ignore_index=True)
    st_df = pd.concat(all_stop_times, ignore_index=True)

    print(f"    Loaded {len(trips_df):,} trips, {len(cal_df)} service patterns, "
          f"{len(st_df):,} stop times")
    return trips_df, cal_df, st_df


def build_gtfs_service_profile(cal_df, st_df, trips_df):
    print("  Building GTFS service profile by day of week...")

    cal_df = cal_df.copy()
    cal_df['start_date'] = pd.to_datetime(cal_df['start_date'], format='%Y%m%d', errors='coerce')
    cal_df['end_date'] = pd.to_datetime(cal_df['end_date'], format='%Y%m%d', errors='coerce')

    day_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    cal_df['weekday_services'] = cal_df[day_cols[:5]].sum(axis=1)
    cal_df['weekend_services'] = cal_df[day_cols[5:]].sum(axis=1)
    cal_df['total_weekly_services'] = cal_df[day_cols].sum(axis=1)

    cal_df['is_weekday_service'] = (cal_df['weekday_services'] > 0).astype(int)
    cal_df['is_weekend_service'] = (cal_df['weekend_services'] > 0).astype(int)
    cal_df['is_daily_service'] = (cal_df['total_weekly_services'] == 7).astype(int)

    trips_service = trips_df.merge(cal_df[['service_id', 'source', 'monday', 'tuesday',
                                            'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                                            'weekday_services', 'weekend_services',
                                            'is_weekday_service', 'is_weekend_service', 'is_daily_service']],
                                   on=['service_id', 'source'], how='left')

    trips_per_service = trips_service.groupby(['service_id', 'source']).size().reset_index(name='trip_count')

    cal_with_trips = cal_df.merge(trips_per_service, on=['service_id', 'source'], how='left')
    cal_with_trips['trip_count'] = cal_with_trips['trip_count'].fillna(0)

    cal_with_trips['weekday_trips'] = (
        cal_with_trips['trip_count'] * cal_with_trips['is_weekday_service'] / cal_with_trips['total_weekly_services'].replace(0, 1)
    )
    cal_with_trips['weekend_trips'] = (
        cal_with_trips['trip_count'] * cal_with_trips['is_weekend_service'] / cal_with_trips['total_weekly_services'].replace(0, 1)
    )

    service_summary = cal_with_trips.groupby('source').agg({
        'service_id': 'count',
        'trip_count': 'sum',
        'weekday_trips': 'sum',
        'weekend_trips': 'sum',
        'is_weekday_service': 'sum',
        'is_weekend_service': 'sum',
        'is_daily_service': 'sum'
    }).reset_index()
    service_summary.columns = ['source', 'n_services', 'total_trips',
                                'weekday_trips', 'weekend_trips',
                                'n_weekday_services', 'n_weekend_services', 'n_daily_services']

    service_summary['weekday_trip_share'] = (
        service_summary['weekday_trips'] / service_summary['total_trips'].replace(0, 1) * 100
    )
    service_summary['weekend_trip_share'] = (
        service_summary['weekend_trips'] / service_summary['total_trips'].replace(0, 1) * 100
    )

    service_summary.to_csv(os.path.join(OUTPUT_DIR, 'holiday_gtfs_service_profile.csv'), index=False)
    print(f"    Saved holiday_gtfs_service_profile.csv")
    print(service_summary)

    return service_summary


def build_holiday_calendar(holidays, ridership_dates):
    print("  Building holiday calendar...")

    holiday_dates = set()
    pre_holiday_dates = set()
    post_holiday_dates = set()

    for _, row in holidays.iterrows():
        start = pd.Timestamp(row['start_date'])
        end = pd.Timestamp(row['end_date'])
        date_range = pd.date_range(start=start, end=end, freq='D')
        holiday_dates.update(date_range)

        for d in date_range:
            if d - pd.Timedelta(days=1) >= ridership_dates.min():
                pre_holiday_dates.add(d - pd.Timedelta(days=1))
            if d + pd.Timedelta(days=1) <= ridership_dates.max():
                post_holiday_dates.add(d + pd.Timedelta(days=1))

    idx = ridership_dates
    cal = pd.DataFrame(index=idx)
    cal['is_holiday'] = cal.index.isin(holiday_dates).astype(int)
    cal['is_pre_holiday'] = cal.index.isin(pre_holiday_dates).astype(int)
    cal['is_post_holiday'] = cal.index.isin(post_holiday_dates).astype(int)
    cal['is_holiday_period'] = ((cal['is_holiday'] == 1) |
                                 (cal['is_pre_holiday'] == 1) |
                                 (cal['is_post_holiday'] == 1)).astype(int)

    for _, row in holidays.iterrows():
        start = pd.Timestamp(row['start_date'])
        end = pd.Timestamp(row['end_date'])
        date_range = pd.date_range(start=start, end=end, freq='D')
        mask = cal.index.isin(date_range)
        cal.loc[mask, 'holiday_type'] = row['type']
        cal.loc[mask, 'event_name'] = row['event_name']
        cal.loc[mask, 'duration_days'] = row['duration_days']

    cal['holiday_type'] = cal['holiday_type'].fillna('Normal')
    cal['event_name'] = cal['event_name'].fillna('Normal')
    cal['duration_days'] = cal['duration_days'].fillna(0)

    cal['day_of_week'] = cal.index.dayofweek
    cal['is_weekend'] = cal['day_of_week'] >= 5
    cal['day_name'] = cal.index.day_name()

    cal['period'] = 'Normal'
    cal.loc[cal['is_pre_holiday'] == 1, 'period'] = 'Pre-Holiday'
    cal.loc[cal['is_holiday'] == 1, 'period'] = 'Holiday'
    cal.loc[cal['is_post_holiday'] == 1, 'period'] = 'Post-Holiday'

    return cal


def analyze_service_demand_alignment(ridership, holiday_cal, service_summary):
    print("  Analyzing service-demand alignment...")

    merged = ridership.join(holiday_cal, how='inner')
    merged = merged.dropna(subset=['total_ridership'])

    ridership_by_period_dow = merged.groupby(['period', 'day_of_week'])['total_ridership'].agg(
        ['mean', 'std', 'count']).reset_index()

    ridership_by_period = merged.groupby('period')['total_ridership'].agg(
        ['mean', 'std', 'count']).reset_index()

    ridership_by_holiday_type = merged.groupby('holiday_type')['total_ridership'].agg(
        ['mean', 'std', 'count']).reset_index()

    service_weekday = service_summary['weekday_trip_share'].mean()
    service_weekend = service_summary['weekend_trip_share'].mean()

    ridership_weekday = merged[merged['is_weekend'] == 0]['total_ridership'].mean()
    ridership_weekend = merged[merged['is_weekend'] == 1]['total_ridership'].mean()
    ridership_holiday = merged[merged['is_holiday'] == 1]['total_ridership'].mean()
    ridership_normal = merged[merged['is_holiday'] == 0]['total_ridership'].mean()

    alignment_stats = pd.DataFrame([{
        'metric': 'Weekday ridership (normal)',
        'value': ridership_weekday,
        'service_weekday_share': service_weekday,
        'service_weekend_share': service_weekend,
        'gap': ridership_weekday - ridership_normal
    }, {
        'metric': 'Weekend ridership (normal)',
        'value': ridership_weekend,
        'service_weekday_share': service_weekday,
        'service_weekend_share': service_weekend,
        'gap': ridership_weekend - ridership_normal
    }, {
        'metric': 'Holiday ridership',
        'value': ridership_holiday,
        'service_weekday_share': service_weekday,
        'service_weekend_share': service_weekend,
        'gap': ridership_holiday - ridership_normal
    }, {
        'metric': 'Normal day ridership',
        'value': ridership_normal,
        'service_weekday_share': service_weekday,
        'service_weekend_share': service_weekend,
        'gap': 0
    }])
    alignment_stats.to_csv(os.path.join(OUTPUT_DIR, 'holiday_gtfs_alignment.csv'), index=False)
    print(f"    Saved holiday_gtfs_alignment.csv")

    ridership_by_period_dow.to_csv(os.path.join(OUTPUT_DIR, 'holiday_gtfs_ridership_by_period_dow.csv'), index=False)

    return merged, alignment_stats, ridership_by_period_dow


def plot_results(merged, alignment_stats, ridership_by_period_dow, service_summary):
    print("  Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    service_melt = service_summary.melt(id_vars='source',
                                        value_vars=['weekday_trip_share', 'weekend_trip_share'],
                                        var_name='trip_type', value_name='share')
    service_melt['trip_type'] = service_melt['trip_type'].map({
        'weekday_trip_share': 'Weekday Trips', 'weekend_trip_share': 'Weekend Trips'
    })
    pivot_svc = service_summary.set_index('source')[['weekday_trip_share', 'weekend_trip_share']]
    pivot_svc.plot(kind='bar', ax=ax, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    ax.set_xlabel('GTFS Source')
    ax.set_ylabel('Trip Share (%)')
    ax.set_title('GTFS Service Trip Share: Weekday vs Weekend')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(['Weekday Trips', 'Weekend Trips'])
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 1]
    period_order = ['Normal', 'Pre-Holiday', 'Holiday', 'Post-Holiday']
    merged_valid = merged[merged['period'].isin(period_order)]
    period_stats = merged_valid.groupby('period')['total_ridership'].agg(['mean', 'std']).reindex(period_order)
    bars = ax.bar(period_order, period_stats['mean'],
                  yerr=period_stats['std'], capsize=5,
                  color=['seagreen', 'orange', 'coral', 'steelblue'], alpha=0.8)
    ax.set_xlabel('Period')
    ax.set_ylabel('Mean Total Ridership')
    ax.set_title('Ridership by Holiday Period')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 2]
    holiday_types = merged[merged['holiday_type'] != 'Normal']['holiday_type'].unique()
    holiday_ridership = merged[merged['holiday_type'] != 'Normal'].groupby('holiday_type')['total_ridership'].mean()
    holiday_ridership = holiday_ridership.sort_values(ascending=True)
    if len(holiday_ridership) > 0:
        ax.barh(holiday_ridership.index, holiday_ridership.values,
                color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(merged['total_ridership'].mean(), color='grey', linestyle='--',
               label=f'Overall Mean: {merged["total_ridership"].mean():,.0f}')
    ax.set_xlabel('Mean Ridership')
    ax.set_title('Ridership by Holiday Type')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1, 0]
    dow_order = [0, 1, 2, 3, 4, 5, 6]
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    merged_valid = merged[merged['period'].isin(period_order)].copy()
    dow_profile = merged_valid.groupby(['period', 'day_of_week'])['total_ridership'].mean().unstack(level=0)
    dow_profile = dow_profile.reindex(dow_order)
    dow_profile = dow_profile[period_order]
    dow_profile.plot(kind='line', marker='o', ax=ax, linewidth=2, markersize=6)
    ax.set_xticks(dow_order)
    ax.set_xticklabels(dow_labels)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Mean Ridership')
    ax.set_title('Ridership Profile by Day-of-Week\n(Holiday Period vs Normal)')
    ax.legend(title='Period')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    merged_valid = merged[merged['is_holiday'] == 1].copy()
    if len(merged_valid) > 0:
        merged_valid['month'] = merged_valid.index.month
        monthly_holiday = merged_valid.groupby('month')['total_ridership'].agg(['mean', 'count'])
        ax.bar(monthly_holiday.index, monthly_holiday['mean'],
               color='coral', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Month')
        ax.set_ylabel('Mean Ridership')
        ax.set_title('Ridership During Holiday Periods by Month')
        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 2]
    service_w = service_summary['weekday_trip_share'].mean()
    service_we = service_summary['weekend_trip_share'].mean()
    ridership_w = merged[merged['is_weekend'] == 0]['total_ridership'].mean()
    ridership_we = merged[merged['is_weekend'] == 1]['total_ridership'].mean()
    ridership_hol = merged[merged['is_holiday'] == 1]['total_ridership'].mean()
    ridership_norm = merged[merged['is_holiday'] == 0]['total_ridership'].mean()

    categories = ['Normal\nWeekday', 'Normal\nWeekend', 'Holiday']
    ridership_vals = [ridership_w, ridership_we, ridership_hol]
    service_vals = [service_w, service_we, 50]

    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, ridership_vals, width, label='Ridership', color='steelblue', alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, service_vals, width, label='Service Share (weekday)', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Mean Ridership')
    ax2.set_ylabel('Weekday Service Share (%)')
    ax.set_title('Ridership vs GTFS Service Alignment')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_gtfs_alignment.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved holiday_gtfs_alignment.png")


def main():
    print("=" * 60)
    print("BIVARIATE EDA: Holiday vs GTFS Service Schedule Alignment")
    print("=" * 60)

    holidays = load_holidays()
    ridership = load_ridership()
    trips_df, cal_df, st_df = load_all_gtfs()

    service_summary = build_gtfs_service_profile(cal_df, st_df, trips_df)
    holiday_cal = build_holiday_calendar(holidays, ridership.index)
    merged, alignment_stats, ridership_by_period_dow = analyze_service_demand_alignment(
        ridership, holiday_cal, service_summary)
    plot_results(merged, alignment_stats, ridership_by_period_dow, service_summary)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()