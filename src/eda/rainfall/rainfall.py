import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.interpolate import griddata
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "../../../data/cleaned/rainfall_combined_final.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE_PCOODES = {
    'MY01': 'Johor', 'MY02': 'Kedah', 'MY03': 'Kelantan', 'MY04': 'Melaka',
    'MY05': 'Negeri Sembilan', 'MY06': 'Pahang', 'MY07': 'Perak', 'MY08': 'Perlis',
    'MY09': 'Penang', 'MY10': 'Sabah', 'MY11': 'Sarawak', 'MY12': 'Selangor',
    'MY13': 'Terengganu', 'MY15': 'KL', 'MY17': 'Putrajaya'
}

STATE_COORDS = {
    'Johor': (1.4854, 103.7616), 'Kedah': (6.1186, 100.3681), 'Kelantan': (6.1256, 102.2430),
    'Melaka': (2.1896, 102.2501), 'Negeri Sembilan': (2.7297, 101.9424), 'Pahang': (3.8124, 103.3266),
    'Perak': (4.5937, 101.0848), 'Perlis': (6.4417, 100.2072), 'Penang': (5.4164, 100.3328),
    'Sabah': (5.9788, 116.0753), 'Sarawak': (1.5533, 110.3597), 'Selangor': (3.0738, 101.5183),
    'Terengganu': (4.1103, 103.4251), 'KL': (3.1390, 101.6869), 'Putrajaya': (2.9264, 101.6969)
}

ADM2_COORDS = {
    'MY0101': (1.8573, 103.1324), 'MY0102': (1.4923, 103.6053), 'MY0103': (1.3383, 104.0919),
    'MY0104': (2.0577, 103.3500), 'MY0105': (1.4923, 103.8208), 'MY0107': (1.7303, 103.2500),
    'MY0108': (1.6500, 103.6000), 'MY0109': (1.5500, 103.9000), 'MY0110': (1.4500, 103.9500),
    'MY0201': (5.1400, 100.5200), 'MY0202': (5.3500, 100.2800), 'MY0203': (5.1000, 100.8500),
    'MY0204': (5.2800, 100.4800), 'MY0205': (4.9500, 100.6000), 'MY0206': (5.2500, 100.4900),
    'MY0207': (5.1500, 100.2000), 'MY0208': (4.8800, 100.7200), 'MY0209': (5.1500, 100.4500),
    'MY0211': (4.6000, 100.9500), 'MY0212': (5.3300, 100.1700),
    'MY0301': (6.1200, 102.1000), 'MY0302': (5.9000, 102.0000), 'MY0303': (5.8500, 102.0500),
    'MY0304': (5.9500, 101.9000), 'MY0305': (5.8000, 102.1000), 'MY0306': (5.7500, 102.1500),
    'MY0307': (5.9500, 101.7000), 'MY0308': (5.8500, 101.8500), 'MY0309': (5.9000, 102.2500),
    'MY0310': (5.7500, 101.9500),
    'MY0401': (2.1896, 102.2501),
    'MY0501': (2.7297, 101.9424),
    'MY0601': (3.5000, 103.0000), 'MY0602': (3.8500, 102.7000), 'MY0603': (4.1000, 102.3000),
    'MY0701': (4.6000, 101.0000), 'MY0702': (4.5000, 101.2000), 'MY0703': (5.0000, 100.8500),
    'MY0704': (4.7000, 100.9000), 'MY0705': (4.4000, 101.1000), 'MY0706': (4.8500, 100.7000),
    'MY0707': (5.0500, 100.6500),
    'MY0801': (6.4417, 100.2072), 'MY0803': (6.4000, 100.1500), 'MY0804': (6.3500, 100.2000),
    'MY0805': (6.4000, 100.2800), 'MY0806': (6.4500, 100.1800), 'MY0807': (6.3800, 100.2200),
    'MY0808': (6.3500, 100.2500), 'MY0809': (6.4200, 100.1600), 'MY0810': (6.4600, 100.1900),
    'MY0811': (6.4100, 100.2400),
    'MY0901': (5.4164, 100.3328), 'MY0902': (5.3000, 100.2700), 'MY0903': (5.4500, 100.3800),
    'MY0904': (5.2000, 100.2400), 'MY0905': (5.1500, 100.2800), 'MY0906': (5.3800, 100.2200),
    'MY0907': (5.1000, 100.3000), 'MY0908': (5.2500, 100.2000), 'MY0909': (5.3500, 100.2500),
    'MY0910': (5.4200, 100.3100),
    'MY1001': (5.9788, 116.0753), 'MY1101': (1.5533, 110.3597), 'MY1102': (1.4500, 110.2000),
    'MY1103': (1.5500, 110.2500), 'MY1104': (1.7000, 110.1000), 'MY1105': (1.8000, 110.1500),
    'MY1201': (3.0738, 101.5183), 'MY1202': (3.0500, 101.5500), 'MY1203': (2.9500, 101.6000),
    'MY1204': (3.1500, 101.4500), 'MY1205': (3.0500, 101.6000), 'MY1206': (3.1000, 101.5000),
    'MY1207': (2.9000, 101.6500), 'MY1208': (3.0000, 101.5500), 'MY1209': (3.2000, 101.4500),
    'MY1211': (2.8500, 101.7000), 'MY1212': (2.9500, 101.6000), 'MY1213': (3.1000, 101.5200),
    'MY1214': (2.8500, 101.7500), 'MY1215': (2.8000, 101.7000), 'MY1217': (3.0500, 101.7000),
    'MY1218': (3.2000, 101.5000), 'MY1219': (2.9500, 101.4800), 'MY1220': (3.1000, 101.6500),
    'MY1221': (2.7500, 101.8000), 'MY1222': (2.8500, 101.6000), 'MY1223': (3.0000, 101.7000),
    'MY1224': (3.2500, 101.5500), 'MY1225': (3.1500, 101.6000),
    'MY1305': (4.8000, 103.0500), 'MY1306': (4.9000, 103.1000), 'MY1308': (4.5000, 103.3000),
    'MY1309': (4.6500, 103.2000), 'MY1310': (4.7500, 103.1500), 'MY1311': (4.6500, 103.0500),
    'MY1312': (4.5500, 103.3500), 'MY1313': (4.7000, 103.2500), 'MY1314': (4.8500, 102.9500),
    'MY1315': (4.5000, 103.1000), 'MY1316': (4.3000, 103.4000), 'MY1317': (4.7000, 103.0000),
    'MY1318': (4.9000, 103.4000), 'MY1319': (5.0000, 103.0500), 'MY1320': (4.9000, 103.2000),
    'MY1322': (4.7500, 103.3000), 'MY1323': (5.1000, 103.0000), 'MY1324': (4.5500, 103.2000),
    'MY1326': (4.9500, 103.1500), 'MY1327': (4.8000, 103.3500), 'MY1328': (4.7000, 103.1000),
    'MY1329': (4.8500, 103.2500), 'MY1330': (4.6000, 103.3000), 'MY1331': (4.8500, 103.3500),
    'MY1501': (3.1000, 101.7000), 'MY1502': (3.2000, 101.6000), 'MY1503': (2.9000, 101.8000),
    'MY1504': (3.3500, 101.6000), 'MY1505': (3.0500, 101.7500), 'MY1506': (2.9500, 101.7000),
    'MY1507': (3.3000, 101.6500), 'MY1701': (2.9264, 101.6969), 'MY1702': (2.9500, 101.6800),
    'MY1703': (2.9000, 101.7200), 'MY1704': (2.8500, 101.7300), 'MY1705': (2.9500, 101.7000),
    'MY1706': (2.9000, 101.7500), 'MY1707': (3.0000, 101.6500), 'MY1708': (2.8500, 101.7000),
    'MY1709': (2.9000, 101.6800)
}

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['month_name'] = df['date'].dt.month_name()

adm1 = df[df['adm_level'] == 1].copy()
adm2 = df[df['adm_level'] == 2].copy()

adm1['state_name'] = adm1['PCODE'].map(STATE_PCOODES)
adm1['lat'] = adm1['state_name'].map(lambda x: STATE_COORDS.get(x, (None, None))[0])
adm1['lon'] = adm1['state_name'].map(lambda x: STATE_COORDS.get(x, (None, None))[1])

def plot_temporal_distribution():
    plt.figure(figsize=(14, 6))
    daily_avg = df.groupby('date')['rainfall_mm'].mean()
    plt.plot(daily_avg.index, daily_avg.values, linewidth=0.8, alpha=0.8)
    plt.title('Daily Average Rainfall Over Time (Malaysia)')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_daily_trend.png'), dpi=150)
    plt.close()

    monthly_avg = df.groupby(['year', 'month'])['rainfall_mm'].mean().reset_index()
    monthly_avg['year_month'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))
    plt.figure(figsize=(14, 6))
    plt.plot(monthly_avg['year_month'], monthly_avg['rainfall_mm'], marker='o', markersize=4)
    plt.title('Monthly Average Rainfall Over Time')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_monthly_trend.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    monthly_clim = df.groupby('month')['rainfall_mm'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(month_names, monthly_clim.values, color='steelblue', edgecolor='navy', alpha=0.8)
    plt.title('Climatological Monthly Rainfall Distribution')
    plt.xlabel('Month')
    plt.ylabel('Average Rainfall (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_monthly_seasonality.png'), dpi=150)
    plt.close()

    monthly_by_year = df.groupby(['year', 'month'])['rainfall_mm'].mean().unstack(level=0)
    plt.figure(figsize=(14, 7))
    monthly_by_year.plot(kind='line', marker='o', markersize=4, alpha=0.7)
    plt.title('Monthly Rainfall by Year')
    plt.xlabel('Month')
    plt.ylabel('Rainfall (mm)')
    plt.legend(title='Year', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_yearly_comparison.png'), dpi=150)
    plt.close()

    yearly_total = df.groupby('year')['rainfall_mm'].mean()
    plt.figure(figsize=(10, 5))
    yearly_total.plot(kind='bar', color='steelblue', edgecolor='navy', alpha=0.8)
    plt.title('Annual Mean Daily Rainfall')
    plt.xlabel('Year')
    plt.ylabel('Mean Rainfall (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_annual.png'), dpi=150)
    plt.close()

    ts = daily_avg.dropna()
    if len(ts) >= 365:
        period = 36
    else:
        period = max(3, len(ts) // 8)
    decomposition = seasonal_decompose(ts, model='additive', period=period)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    plt.suptitle('Time Series Decomposition (Additive)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_decomposition.png'), dpi=150)
    plt.close()

    print("Temporal distribution plots saved.")

def plot_spatial_analysis():
    state_avg = adm1.groupby('state_name').agg({
        'rainfall_mm': 'mean',
        'anomaly_rf': 'mean'
    }).reset_index()

    state_avg['lat'] = state_avg['state_name'].map(lambda x: STATE_COORDS.get(x, (None, None))[0])
    state_avg['lon'] = state_avg['state_name'].map(lambda x: STATE_COORDS.get(x, (None, None))[1])
    state_avg = state_avg.dropna(subset=['lat', 'lon'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    scatter1 = axes[0].scatter(state_avg['lon'], state_avg['lat'],
                                c=state_avg['rainfall_mm'], cmap='Blues',
                                s=200, edgecolor='navy', alpha=0.8)
    for _, row in state_avg.iterrows():
        axes[0].annotate(row['state_name'], (row['lon'], row['lat']),
                        fontsize=7, ha='center', va='bottom')
    axes[0].set_title('Mean Rainfall by State (mm)')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(scatter1, ax=axes[0], label='Rainfall (mm)')

    scatter2 = axes[1].scatter(state_avg['lon'], state_avg['lat'],
                                c=state_avg['anomaly_rf'], cmap='RdBu_r',
                                s=200, edgecolor='black', alpha=0.8)
    for _, row in state_avg.iterrows():
        axes[1].annotate(row['state_name'], (row['lon'], row['lat']),
                        fontsize=7, ha='center', va='bottom')
    axes[1].set_title('Mean Rainfall Anomaly by State')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(scatter2, ax=axes[1], label='Anomaly (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'spatial_state_scatter.png'), dpi=150)
    plt.close()

    lon_range = np.linspace(99.5, 104.5, 50)
    lat_range = np.linspace(0.5, 7.5, 50)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

    if len(state_avg) >= 4:
        points = state_avg[['lon', 'lat']].values
        values = state_avg['rainfall_mm'].values
        try:
            rainfall_grid = griddata(points, values, (lon_grid, lat_grid), method='cubic', fill_value=np.nan)
        except:
            rainfall_grid = griddata(points, values, (lon_grid, lat_grid), method='linear', fill_value=np.nan)

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.contourf(lon_grid, lat_grid, rainfall_grid, levels=20, cmap='Blues')
        ax.scatter(state_avg['lon'], state_avg['lat'], c='red', s=50, edgecolor='white', zorder=5)
        for _, row in state_avg.iterrows():
            ax.annotate(row['state_name'], (row['lon'], row['lat']),
                       fontsize=7, ha='center', va='bottom', color='darkred')
        ax.set_title('Interpolated Mean Rainfall Heatmap (Malaysia)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label='Rainfall (mm)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'spatial_heatmap_interpolated.png'), dpi=150)
        plt.close()
    else:
        print("Insufficient state data for interpolation.")

    adm2_sample = adm2[adm2['PCODE'].isin(ADM2_COORDS.keys())].copy()
    if len(adm2_sample) > 0:
        adm2_sample['lat'] = adm2_sample['PCODE'].map(lambda x: ADM2_COORDS.get(x, (None, None))[0])
        adm2_sample['lon'] = adm2_sample['PCODE'].map(lambda x: ADM2_COORDS.get(x, (None, None))[1])
        adm2_avg = adm2_sample.groupby('PCODE').agg({
            'rainfall_mm': 'mean', 'lat': 'first', 'lon': 'first'
        }).reset_index()
        adm2_avg = adm2_avg.dropna()

        if len(adm2_avg) >= 4:
            plt.figure(figsize=(14, 8))
            scatter = plt.scatter(adm2_avg['lon'], adm2_avg['lat'],
                                 c=adm2_avg['rainfall_mm'], cmap='Blues',
                                 s=100, edgecolor='navy', alpha=0.8)
            plt.title('Mean Rainfall by District (ADM2)')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.colorbar(scatter, label='Rainfall (mm)')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'spatial_adm2_scatter.png'), dpi=150)
            plt.close()

            lon_range2 = np.linspace(99.5, 104.5, 80)
            lat_range2 = np.linspace(0.5, 7.5, 80)
            lon_grid2, lat_grid2 = np.meshgrid(lon_range2, lat_range2)

            points2 = adm2_avg[['lon', 'lat']].values
            values2 = adm2_avg['rainfall_mm'].values
            try:
                rainfall_grid2 = griddata(points2, values2, (lon_grid2, lat_grid2), method='cubic', fill_value=np.nan)
            except:
                rainfall_grid2 = griddata(points2, values2, (lon_grid2, lat_grid2), method='linear', fill_value=np.nan)

            fig, ax = plt.subplots(figsize=(14, 9))
            im = ax.contourf(lon_grid2, lat_grid2, rainfall_grid2, levels=30, cmap='Blues')
            ax.scatter(adm2_avg['lon'], adm2_avg['lat'], c='red', s=30, edgecolor='white', zorder=5)
            ax.set_title('Interpolated Mean Rainfall Heatmap (District Level)')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, label='Rainfall (mm)')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'spatial_adm2_heatmap.png'), dpi=150)
            plt.close()

    monthly_spatial = adm1.groupby(['month', 'state_name']).agg({
        'rainfall_mm': 'mean',
        'lat': 'first',
        'lon': 'first'
    }).reset_index()

    months_to_plot = [1, 4, 7, 10]
    month_names_map = {1: 'January', 4: 'April', 7: 'July', 10: 'October'}
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, month in enumerate(months_to_plot):
        month_data = monthly_spatial[monthly_spatial['month'] == month].dropna(subset=['lat', 'lon'])
        if len(month_data) > 0:
            scatter = axes[i].scatter(month_data['lon'], month_data['lat'],
                                     c=month_data['rainfall_mm'], cmap='Blues',
                                     s=150, edgecolor='navy', alpha=0.8)
            for _, row in month_data.iterrows():
                axes[i].annotate(row['state_name'][:3], (row['lon'], row['lat']),
                               fontsize=6, ha='center', va='bottom')
            axes[i].set_title(f'{month_names_map[month]}')
            axes[i].set_xlabel('Longitude')
            axes[i].set_ylabel('Latitude')
            plt.colorbar(scatter, ax=axes[i], label='Rainfall (mm)')

    plt.suptitle('Spatial Rainfall Distribution by Season', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'spatial_seasonal_comparison.png'), dpi=150)
    plt.close()

    print("Spatial analysis plots saved.")

def plot_extreme_events():
    wet_threshold = 10.0
    very_wet_threshold = 30.0
    extreme_threshold = 50.0

    df['is_wet'] = df['rainfall_mm'] >= wet_threshold
    df['is_very_wet'] = df['rainfall_mm'] >= very_wet_threshold
    df['is_extreme'] = df['rainfall_mm'] >= extreme_threshold

    df['year_month'] = df['date'].dt.to_period('M')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    extreme_monthly = df.groupby('year_month').agg({
        'is_extreme': 'sum',
        'is_very_wet': 'sum',
        'is_wet': 'sum',
        'date': 'count'
    }).rename(columns={'date': 'total_records'})
    extreme_monthly['extreme_pct'] = (extreme_monthly['is_extreme'] / extreme_monthly['total_records']) * 100

    extreme_monthly.index = extreme_monthly.index.to_timestamp()
    axes[0, 0].plot(extreme_monthly.index, extreme_monthly['extreme_pct'], marker='o', markersize=3, color='crimson')
    axes[0, 0].set_title('Monthly Extreme Rainfall Event Frequency (%)')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Extreme Days (%)')

    extreme_yearly = df.groupby('year').agg({
        'is_extreme': 'sum',
        'is_very_wet': 'sum',
        'is_wet': 'sum',
        'date': 'count'
    }).rename(columns={'date': 'total_records'})
    extreme_yearly['extreme_pct'] = (extreme_yearly['is_extreme'] / extreme_yearly['total_records']) * 100

    axes[0, 1].bar(extreme_yearly.index, extreme_yearly['extreme_pct'], color='crimson', alpha=0.8, edgecolor='darkred')
    axes[0, 1].set_title('Annual Extreme Rainfall Event Frequency (%)')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Extreme Days (%)')

    axes[1, 0].bar(extreme_yearly.index, extreme_yearly['is_very_wet'], color='orange', alpha=0.8, label='Very Wet (>=30mm)')
    axes[1, 0].bar(extreme_yearly.index, extreme_yearly['is_extreme'], color='crimson', alpha=0.8, label='Extreme (>=50mm)')
    axes[1, 0].set_title('Annual Wet Day Counts by Category')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Number of Days')
    axes[1, 0].legend()

    monthly_extreme = df.groupby('month').agg({
        'is_wet': 'sum',
        'is_very_wet': 'sum',
        'is_extreme': 'sum'
    })
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x = np.arange(len(month_names))
    width = 0.25
    axes[1, 1].bar(x - width, monthly_extreme['is_wet'], width, label='Wet (>=10mm)', color='steelblue', alpha=0.8)
    axes[1, 1].bar(x, monthly_extreme['is_very_wet'], width, label='Very Wet (>=30mm)', color='orange', alpha=0.8)
    axes[1, 1].bar(x + width, monthly_extreme['is_extreme'], width, label='Extreme (>=50mm)', color='crimson', alpha=0.8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(month_names)
    axes[1, 1].set_title('Extreme Event Frequency by Month')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Number of Days')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'extreme_events_frequency.png'), dpi=150)
    plt.close()

    adm1_events = df[df['adm_level'] == 1].copy()
    adm1_events['state_name'] = adm1_events['PCODE'].map(STATE_PCOODES)
    state_extreme = adm1_events.groupby('state_name').agg({
        'is_wet': 'sum',
        'is_very_wet': 'sum',
        'is_extreme': 'sum'
    }).reset_index()
    state_extreme = state_extreme.sort_values('is_extreme', ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].barh(state_extreme['state_name'], state_extreme['is_wet'], color='steelblue', alpha=0.8)
    axes[0].set_title('Wet Days (>=10mm) by State')
    axes[0].set_xlabel('Count')

    axes[1].barh(state_extreme['state_name'], state_extreme['is_very_wet'], color='orange', alpha=0.8)
    axes[1].set_title('Very Wet Days (>=30mm) by State')
    axes[1].set_xlabel('Count')

    axes[2].barh(state_extreme['state_name'], state_extreme['is_extreme'], color='crimson', alpha=0.8)
    axes[2].set_title('Extreme Days (>=50mm) by State')
    axes[2].set_xlabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'extreme_events_by_state.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.histplot(df['rainfall_mm'].dropna(), bins=50, color='steelblue', alpha=0.7, kde=False)
    plt.axvline(x=wet_threshold, color='blue', linestyle='--', label=f'Wet (>=10mm)')
    plt.axvline(x=very_wet_threshold, color='orange', linestyle='--', label=f'Very Wet (>=30mm)')
    plt.axvline(x=extreme_threshold, color='crimson', linestyle='--', label=f'Extreme (>=50mm)')
    plt.title('Distribution of Daily Rainfall with Thresholds')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'extreme_events_distribution.png'), dpi=150)
    plt.close()

    print("Extreme event frequency analysis plots saved.")

def plot_wet_dry_ratio():
    df['is_wet'] = df['rainfall_mm'] >= 1.0

    monthly_stats = df.groupby('month').agg(
        total_days=('rainfall_mm', 'count'),
        wet_days=('is_wet', 'sum')
    )
    monthly_stats['dry_days'] = monthly_stats['total_days'] - monthly_stats['wet_days']
    monthly_stats['wet_ratio'] = monthly_stats['wet_days'] / monthly_stats['total_days']
    monthly_stats['dry_ratio'] = monthly_stats['dry_days'] / monthly_stats['total_days']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x = np.arange(len(month_names))
    width = 0.35

    axes[0, 0].bar(x - width/2, monthly_stats['wet_days'], width, label='Wet Days', color='steelblue', alpha=0.8)
    axes[0, 0].bar(x + width/2, monthly_stats['dry_days'], width, label='Dry Days', color='sandybrown', alpha=0.8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(month_names)
    axes[0, 0].set_title('Wet vs Dry Days by Month')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Number of Days')
    axes[0, 0].legend()

    axes[0, 1].plot(month_names, monthly_stats['wet_ratio'], marker='o', color='steelblue', linewidth=2, label='Wet Ratio')
    axes[0, 1].plot(month_names, monthly_stats['dry_ratio'], marker='s', color='sandybrown', linewidth=2, label='Dry Ratio')
    axes[0, 1].set_title('Wet/Dry Day Ratio by Month')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Ratio')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].bar(month_names, monthly_stats['wet_ratio'] * 100, color='steelblue', alpha=0.8, edgecolor='navy')
    axes[1, 0].set_title('Wet Day Percentage by Month')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Wet Day (%)')
    axes[1, 0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% line')
    axes[1, 0].legend()

    year_month_stats = df.groupby(['year', 'month']).agg(
        total_days=('rainfall_mm', 'count'),
        wet_days=('is_wet', 'sum')
    ).reset_index()
    year_month_stats['wet_ratio'] = year_month_stats['wet_days'] / year_month_stats['total_days']
    year_month_stats['month_name'] = year_month_stats['month'].apply(lambda m: month_names[m-1])
    year_month_stats_pivot = year_month_stats.pivot(index='year', columns='month_name', values='wet_ratio')
    year_month_stats_pivot = year_month_stats_pivot[month_names]
    year_month_stats_pivot.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='Blues', alpha=0.8, width=0.7)
    axes[1, 1].set_title('Wet Day Ratio by Year and Month')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Wet Ratio')
    axes[1, 1].legend(title='Month', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wet_dry_ratio_monthly.png'), dpi=150)
    plt.close()

    adm1_wet = df[df['adm_level'] == 1].copy()
    adm1_wet['state_name'] = adm1_wet['PCODE'].map(STATE_PCOODES)
    state_stats = adm1_wet.groupby('state_name').agg(
        total_days=('rainfall_mm', 'count'),
        wet_days=('is_wet', 'sum')
    ).reset_index()
    state_stats['wet_ratio'] = state_stats['wet_days'] / state_stats['total_days']
    state_stats = state_stats.sort_values('wet_ratio', ascending=False)

    state_lat = state_stats['state_name'].map(lambda x: STATE_COORDS.get(x, (None, None))[0])
    state_lon = state_stats['state_name'].map(lambda x: STATE_COORDS.get(x, (None, None))[1])
    state_stats['lat'] = state_lat
    state_stats['lon'] = state_lon
    state_stats = state_stats.dropna(subset=['lat', 'lon'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(state_stats['state_name'], state_stats['wet_ratio'] * 100, color='steelblue', alpha=0.8)
    axes[0].set_title('Wet Day Ratio by State')
    axes[0].set_xlabel('Wet Day (%)')
    axes[0].axvline(x=50, color='red', linestyle='--', alpha=0.7)

    scatter = axes[1].scatter(state_stats['lon'], state_stats['lat'],
                            c=state_stats['wet_ratio'] * 100, cmap='Blues',
                            s=200, edgecolor='navy', alpha=0.8)
    for _, row in state_stats.iterrows():
        axes[1].annotate(row['state_name'][:3], (row['lon'], row['lat']),
                        fontsize=7, ha='center', va='bottom')
    axes[1].set_title('Wet Day Ratio by State (Spatial)')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(scatter, ax=axes[1], label='Wet Day (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wet_dry_ratio_state.png'), dpi=150)
    plt.close()

    monthly_by_year = df.groupby(['year', 'month']).agg(
        total_days=('rainfall_mm', 'count'),
        wet_days=('is_wet', 'sum')
    ).reset_index()
    monthly_by_year['wet_ratio'] = monthly_by_year['wet_days'] / monthly_by_year['total_days']
    monthly_pivot = monthly_by_year.pivot(index='year', columns='month', values='wet_ratio')

    plt.figure(figsize=(12, 6))
    sns.heatmap(monthly_pivot, annot=True, fmt='.2f', cmap='Blues', linewidths=0.5,
                xticklabels=month_names, cbar_kws={'label': 'Wet Ratio'})
    plt.title('Wet Day Ratio Heatmap (Year x Month)')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wet_dry_ratio_heatmap.png'), dpi=150)
    plt.close()

    df['wet_category'] = pd.cut(df['rainfall_mm'],
                                bins=[-0.1, 0.1, 10, 30, 50, 100, 500],
                                labels=['Dry (0)', 'Light (0.1-10)', 'Moderate (10-30)',
                                       'Heavy (30-50)', 'Very Heavy (50-100)', 'Extreme (>100)'])

    category_counts = df['wet_category'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['sandybrown', 'lightblue', 'steelblue', 'orange', 'darkorange', 'crimson']
    category_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='navy', alpha=0.8)
    axes[0].set_title('Distribution of Wet/Dry Categories')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    category_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=colors,
                         startangle=90, explode=[0.05]*len(category_counts))
    axes[1].set_title('Proportion of Wet/Dry Categories')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wet_dry_category_distribution.png'), dpi=150)
    plt.close()

    print("Wet/dry ratio analysis plots saved.")

if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Starting Rainfall EDA...")
    plot_temporal_distribution()
    plot_spatial_analysis()
    plot_extreme_events()
    plot_wet_dry_ratio()
    print(f"\nAll EDA plots saved to '{OUTPUT_DIR}' directory.")