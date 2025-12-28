# Unemployment Analysis with Python
# CodeAlpha Data Science Internship - Task 2

import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Download dataset using kagglehub
path = kagglehub.dataset_download("gokulrajkmv/unemployment-in-india")
print("Dataset path:", path)

# Find CSV file automatically
csv_file = None
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)

# Load dataset
df = pd.read_csv(csv_file)

# Clean column names
df.columns = df.columns.str.strip()

print("\nDataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Handle missing values
df.dropna(inplace=True)

# ===============================
# EXPLORATORY DATA ANALYSIS (EDA)
# ===============================

# Average unemployment by region
region_avg = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(10, 6))
region_avg.sort_values().plot(kind="barh", color="skyblue")
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Region")
plt.show()

# ===============================
# TIME SERIES TREND
# ===============================

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Date", y="Estimated Unemployment Rate (%)", hue="Region", legend=False)
plt.title("Unemployment Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# ===============================
# COVID-19 IMPACT ANALYSIS
# ===============================

df["Covid_Period"] = np.where(df["Date"] >= "2020-03-01", "Post-Covid", "Pre-Covid")

covid_avg = df.groupby("Covid_Period")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(6, 4))
covid_avg.plot(kind="bar", color=["green", "red"])
plt.title("Unemployment Rate: Pre vs Post Covid")
plt.ylabel("Unemployment Rate (%)")
plt.show()

print("\nAverage Unemployment Rate (Pre vs Post Covid):")
print(covid_avg)

# ===============================
# SEASONAL ANALYSIS
# ===============================

df["Month"] = df["Date"].dt.month
monthly_avg = df.groupby("Month")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(10, 5))
monthly_avg.plot(marker="o")
plt.title("Seasonal Trend in Unemployment")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()

# ===============================
# KEY INSIGHTS
# ===============================

print("\nKey Insights:")
print("- Unemployment increased sharply after Covid-19.")
print("- Certain regions show consistently higher unemployment.")
print("- Seasonal trends are visible across months.")
print("- Insights can support employment and policy planning.")
