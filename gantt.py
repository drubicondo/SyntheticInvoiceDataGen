import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Task data
tasks = [
    {"Task": "Synthetic Data Generator", "Start": "2025-06-30", "End": "2025-07-11"},
    {"Task": "Graph Construction", "Start": "2025-07-21", "End": "2025-07-25"},
    {"Task": "NLP Embeddings", "Start": "2025-07-27", "End": "2025-08-01"},
    {"Task": "Feature Engineering", "Start": "2025-08-04", "End": "2025-08-08"},
    {"Task": "Modelling", "Start": "2025-08-25", "End": "2025-09-05"},
    {"Task": "Fine-Tuning & Predictions", "Start": "2025-09-15", "End": "2025-09-26"}
]

# Convert to DataFrame
df = pd.DataFrame(tasks)
df["Start"] = pd.to_datetime(df["Start"])
df["End"] = pd.to_datetime(df["End"])
df["Duration"] = (df["End"] - df["Start"]).dt.days

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for i, task in df.iterrows():
    ax.barh(task["Task"], task["Duration"], left=task["Start"], color="skyblue")

# Highlight holidays
holiday_ranges = [
    ("2025-07-12", "2025-07-20"),
    ("2025-08-09", "2025-08-24"),
    ("2025-09-08", "2025-09-12")
]
for start, end in holiday_ranges:
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="lightgrey", alpha=0.5)

# Formatting
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
plt.xticks(rotation=45)
ax.set_title("Project Gantt Chart with Holidays")
ax.set_xlabel("Date")
ax.set_ylabel("Tasks")
plt.tight_layout()

plt.show()