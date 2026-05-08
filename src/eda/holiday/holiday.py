#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Holiday Calendar Data
Objectives:
1. Holiday calendar frequency & spread
2. School holiday vs public holiday overlap
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.max_columns = None


def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['start_date', 'end_date'])
    return df


def frequency_by_type(df):
    """Analyze frequency of holidays by type (Academic vs Public)."""
    type_counts = df['type'].value_counts()
    print("\n=== Holiday Frequency by Type ===")
    print(type_counts)

    plt.figure(figsize=(10, 5))
    type_counts.plot(kind='bar', color=['#2ecc71', '#3498db'])
    plt.title('Holiday Count by Type')
    plt.xlabel('Holiday Type')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_frequency_by_type.png'))
    plt.close()

    return type_counts


def frequency_by_year(df):
    """Analyze holiday frequency spread across years."""
    year_type_counts = df.groupby(['year', 'type']).size().unstack(fill_value=0)
    print("\n=== Holiday Frequency by Year ===")
    print(year_type_counts)

    plt.figure(figsize=(14, 6))
    year_type_counts.plot(kind='bar', stacked=True, color=['#2ecc71', '#3498db'])
    plt.title('Holiday Count by Year and Type')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(title='Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_frequency_by_year.png'))
    plt.close()

    return year_type_counts


def duration_analysis(df):
    """Analyze duration distribution by holiday type."""
    print("\n=== Duration Statistics by Type ===")
    duration_stats = df.groupby('type')['duration_days'].describe()
    print(duration_stats)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='type', y='duration_days')
    plt.title('Duration Distribution by Type')
    plt.xlabel('Holiday Type')
    plt.ylabel('Duration (Days)')

    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x='duration_days', hue='type', kde=True, bins=20)
    plt.title('Duration Histogram by Type')
    plt.xlabel('Duration (Days)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_duration_analysis.png'))
    plt.close()


def spatial_coverage_analysis(df):
    """Analyze spatial coverage distribution."""
    spatial_counts = df['spatial_coverage'].value_counts()
    print("\n=== Spatial Coverage Distribution ===")
    print(spatial_counts)

    plt.figure(figsize=(12, 6))
    spatial_counts.plot(kind='bar')
    plt.title('Holiday Count by Spatial Coverage')
    plt.xlabel('Spatial Coverage')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_spatial_coverage.png'))
    plt.close()


def overlap_analysis(df):
    """Detect overlaps between Academic and Public holidays."""
    academic = df[df['type'] == 'Academic'].copy()
    public = df[df['type'] == 'Public'].copy()

    overlaps = []
    for _, ac_row in academic.iterrows():
        ac_start = ac_row['start_date']
        ac_end = ac_row['end_date']
        ac_name = ac_row['event_name']
        ac_group = ac_row['spatial_coverage']

        for _, pb_row in public.iterrows():
            pb_start = pb_row['start_date']
            pb_end = pb_row['end_date']
            pb_name = pb_row['event_name']

            if ac_start <= pb_end and pb_start <= ac_end:
                overlap_start = max(ac_start, pb_start)
                overlap_end = min(ac_end, pb_end)
                overlap_days = (overlap_end - overlap_start).days + 1
                overlaps.append({
                    'academic_holiday': ac_name,
                    'public_holiday': pb_name,
                    'academic_group': ac_group,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_days': overlap_days
                })

    overlap_df = pd.DataFrame(overlaps)

    if not overlap_df.empty:
        print("\n=== School-Public Holiday Overlaps ===")
        print(f"Total overlaps found: {len(overlap_df)}")
        print(overlap_df.to_string())

        overlap_df.to_csv(os.path.join(OUTPUT_DIR, 'holiday_overlaps.csv'), index=False)

        plt.figure(figsize=(14, 6))
        overlap_by_year = overlap_df.copy()
        overlap_by_year['year'] = overlap_by_year['overlap_start'].dt.year
        overlap_counts = overlap_by_year.groupby('year').size()

        plt.subplot(1, 2, 1)
        overlap_counts.plot(kind='bar', color='#9b59b6')
        plt.title('Number of Overlaps by Year')
        plt.xlabel('Year')
        plt.ylabel('Count')

        plt.subplot(1, 2, 2)
        top_overlaps = overlap_df.nlargest(10, 'overlap_days')
        labels = top_overlaps.apply(
            lambda x: f"{x['academic_holiday'][:15]}...\nvs\n{x['public_holiday'][:15]}...",
            axis=1
        )
        plt.barh(range(len(top_overlaps)), top_overlaps['overlap_days'], color='#9b59b6')
        plt.yticks(range(len(top_overlaps)), labels)
        plt.title('Top 10 Overlaps by Duration')
        plt.xlabel('Overlap Days')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_overlap_analysis.png'))
        plt.close()
    else:
        print("\n=== No overlaps found ===")

    return overlap_df


def monthly_distribution(df):
    """Analyze which months have the most holidays."""
    df['start_month'] = df['start_date'].dt.month
    df['start_month_name'] = df['start_date'].dt.month_name()

    month_type_counts = df.groupby(['start_month', 'start_month_name', 'type']).size().reset_index(name='count')

    plt.figure(figsize=(14, 6))
    month_order = range(1, 13)
    pivot_data = df.groupby(['start_month', 'type']).size().unstack(fill_value=0)
    pivot_data = pivot_data.reindex(month_order)
    pivot_data.plot(kind='bar', stacked=True, color=['#2ecc71', '#3498db'])
    plt.title('Holiday Distribution by Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
               rotation=0)
    plt.legend(title='Type')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_monthly_distribution.png'))
    plt.close()

    return month_type_counts


def group_a_vs_b_analysis(df):
    """Compare Group A vs Group B holiday patterns."""
    group_data = df[df['spatial_coverage'].isin(['Group A', 'Group B'])]

    print("\n=== Group A vs Group B Analysis ===")
    group_counts = group_data.groupby(['year', 'spatial_coverage']).size().unstack(fill_value=0)
    print(group_counts)

    plt.figure(figsize=(14, 6))
    group_counts.plot(kind='line', marker='o')
    plt.title('Holiday Count: Group A vs Group B Over Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(title='Group')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'group_a_vs_b_comparison.png'))
    plt.close()


def event_name_analysis(df):
    """Analyze recurring vs unique holiday events."""
    event_counts = df['event_name'].value_counts()
    recurring = event_counts[event_counts > 1]

    print("\n=== Recurring Holiday Events ===")
    print(f"Total unique events: {df['event_name'].nunique()}")
    print(f"Recurring events (>1 occurrence): {len(recurring)}")
    print(recurring)

    plt.figure(figsize=(12, 8))
    event_counts.head(20).plot(kind='barh', color='#e74c3c')
    plt.title('Top 20 Most Frequent Holiday Events')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_event_frequency.png'))
    plt.close()


def summary_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("HOLIDAY CALENDAR EDA SUMMARY")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['start_date'].min().date()} to {df['end_date'].max().date()}")
    print(f"Years covered: {df['year'].min()} - {df['year'].max()}")
    print(f"Unique events: {df['event_name'].nunique()}")
    print(f"Academic holidays: {len(df[df['type'] == 'Academic'])}")
    print(f"Public holidays: {len(df[df['type'] == 'Public'])}")
    print(f"Total holiday-days: {df['duration_days'].sum()}")
    print(f"Avg holiday duration: {df['duration_days'].mean():.2f} days")
    print("="*60)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    filepath = os.path.join(project_root, 'data', 'cleaned', 'school_public_holiday_clean.csv')
    df = load_data(filepath)

    print("Running Frequency by Type Analysis...")
    frequency_by_type(df)

    print("Running Frequency by Year Analysis...")
    frequency_by_year(df)

    print("Running Duration Analysis...")
    duration_analysis(df)

    print("Running Spatial Coverage Analysis...")
    spatial_coverage_analysis(df)

    print("Running Overlap Analysis...")
    overlap_df = overlap_analysis(df)

    print("Running Monthly Distribution Analysis...")
    monthly_distribution(df)

    print("Running Group A vs B Analysis...")
    group_a_vs_b_analysis(df)

    print("Running Event Name Analysis...")
    event_name_analysis(df)

    print("Running Summary Statistics...")
    summary_statistics(df)

    print(f"\nEDA completed. Results saved to '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()