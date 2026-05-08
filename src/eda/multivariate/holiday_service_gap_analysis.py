import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).parent.parent.parent.parent / "data" / "cleaned"
OUT = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

ridership = pd.read_csv(BASE / "ridership_headline_clean.csv", parse_dates=["date"])
holidays = pd.read_csv(BASE / "school_public_holiday_clean.csv", parse_dates=["start_date", "end_date"])

stops_dfs = []
for gtfs_dir in ["gtfs_rapid_bus_kl", "gtfs_ktmb", "gtfs_rapid_rail_kl"]:
    stops_file = BASE / gtfs_dir / "stops.txt"
    calendar_file = BASE / gtfs_dir / "calendar.txt"
    trips_file = BASE / gtfs_dir / "trips.txt"
    stop_times_file = BASE / gtfs_dir / "stop_times.txt"

    if stops_file.exists() and calendar_file.exists():
        stops_df = pd.read_csv(stops_file)
        calendar_df = pd.read_csv(calendar_file)
        trips_df = pd.read_csv(trips_file) if trips_file.exists() else pd.DataFrame()
        stop_times_df = pd.read_csv(stop_times_file) if stop_times_file.exists() else pd.DataFrame()

        stops_df["source"] = gtfs_dir
        stops_dfs.append({"stops": stops_df, "calendar": calendar_df, "trips": trips_df, "stop_times": stop_times_df})

service_by_date = {}
for gtfs_data in stops_dfs:
    calendar = gtfs_data["calendar"]
    for _, row in calendar.iterrows():
        service_id = row["service_id"]
        start = str(row["start_date"])
        end = str(row["end_date"])
        try:
            start_dt = pd.to_datetime(start, format="%Y%m%d")
            end_dt = pd.to_datetime(end, format="%Y%m%d")
            dates = pd.date_range(start_dt, end_dt)
            for d in dates:
                if d not in service_by_date:
                    service_by_date[d] = 0
                service_by_date[d] += 1
        except:
            pass

service_df = pd.DataFrame({"date": list(service_by_date.keys()), "service_frequency": list(service_by_date.values())})
service_df["date"] = pd.to_datetime(service_df["date"])

holiday_dates = set()
for _, row in holidays.iterrows():
    daterange = pd.date_range(row["start_date"], row["end_date"])
    holiday_dates.update(daterange)

merged = ridership.copy()
merged["is_holiday"] = merged["date"].isin(holiday_dates).astype(int)
merged = merged.merge(service_df, on="date", how="left")
merged["service_frequency"] = merged["service_frequency"].ffill().bfill()
merged["year"] = merged["date"].dt.year
merged["month"] = merged["date"].dt.month
merged["day_of_week"] = merged["date"].dt.dayofweek

holiday_ridership = merged[merged["is_holiday"] == 1].groupby("date").agg({"total_ridership": "mean", "service_frequency": "mean"}).reset_index()
normal_ridership = merged[merged["is_holiday"] == 0].groupby("date").agg({"total_ridership": "mean", "service_frequency": "mean"}).reset_index()

gap = merged.groupby(["year", "month"]).agg({
    "total_ridership": "mean",
    "service_frequency": "mean",
    "is_holiday": "mean"
}).reset_index()
gap["date"] = pd.to_datetime(gap["year"].astype(str) + "-" + gap["month"].astype(str) + "-01")
gap = gap.sort_values("date")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ax1 = axes[0, 0]
ax1.scatter(merged[merged["is_holiday"] == 0]["service_frequency"], merged[merged["is_holiday"] == 0]["total_ridership"] / 1e6, alpha=0.3, s=20, label="Normal Day", color="gray")
ax1.scatter(merged[merged["is_holiday"] == 1]["service_frequency"], merged[merged["is_holiday"] == 1]["total_ridership"] / 1e6, alpha=0.7, s=40, label="Holiday", color="red")
ax1.set_xlabel("Service Frequency (stops/day)")
ax1.set_ylabel("Ridership (Millions)")
ax1.set_title("Service Frequency vs Ridership: Holiday vs Normal")
ax1.legend()

ax2 = axes[0, 1]
gap_sorted = gap.sort_values("is_holiday", ascending=False).head(20)
ax2.bar(range(len(gap_sorted)), gap_sorted["service_frequency"], color="steelblue", alpha=0.7, label="Service Frequency")
ax2_twin = ax2.twinx()
ax2_twin.plot(range(len(gap_sorted)), gap_sorted["total_ridership"] / 1e6, color="red", marker="o", label="Ridership")
ax2.set_xticks(range(len(gap_sorted)))
ax2.set_xticklabels([f"{r['year']}-{r['month']:02d}" for _, r in gap_sorted.iterrows()], rotation=45, fontsize=7)
ax2.set_ylabel("Service Frequency")
ax2_twin.set_ylabel("Ridership (Millions)", color="red")
ax2.set_title("Holiday Service Gap: Top 20 Holiday Months")

merged["ridership_per_service"] = merged["total_ridership"] / merged["service_frequency"]
gap_analysis = merged.groupby("is_holiday").agg({
    "total_ridership": "mean",
    "service_frequency": "mean",
    "ridership_per_service": "mean"
}).reset_index()
ax3 = axes[1, 0]
x = range(len(gap_analysis))
ax3.bar([i - 0.2 for i in x], gap_analysis["total_ridership"] / 1e6, width=0.4, label="Avg Ridership", color="steelblue")
ax3.bar([i + 0.2 for i in x], gap_analysis["service_frequency"] / 10, width=0.4, label="Service Freq (/10)", color="orange")
ax3.set_xticks(x)
ax3.set_xticklabels(["Normal", "Holiday"])
ax3.set_ylabel("Value")
ax3.set_title("Holiday vs Normal: Ridership & Service")
ax3.legend()

monthly = gap.copy()
monthly = monthly.sort_values("date")
ax4 = axes[1, 1]
ax4.plot(monthly["date"], monthly["service_frequency"], marker="o", color="steelblue", label="Service Frequency")
ax4.plot(monthly["date"], monthly["total_ridership"] / 1e6 * 100, marker="s", color="red", label="Ridership (x100)")
ax4.set_xlabel("Date")
ax4.set_ylabel("Value")
ax4.set_title("Monthly Service Frequency vs Ridership")
ax4.legend()

plt.tight_layout()
plt.savefig(OUT / "holiday_service_gap_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

gap_analysis.to_csv(OUT / "holiday_service_gap.csv", index=False)
merged[["date", "total_ridership", "service_frequency", "is_holiday", "ridership_per_service"]].to_csv(OUT / "holiday_service_detail.csv", index=False)
print("5. holiday_service_gap_analysis.py - DONE")