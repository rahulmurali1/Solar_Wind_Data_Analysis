import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("Energy Production Dataset.csv")

sns.set_style("whitegrid") #sets grid color and theme
#rcParams = runtime configuration parameters
plt.rcParams["figure.facecolor"] = "white" #set the outside background
plt.rcParams["axes.facecolor"] = "white" #sets the inside background

print("Data loaded successfully")

df = df[df['Source'] != "Mixed"] #removed 2 entries where source = Mixed

#Plotting a histogram with KDE
fig, ax = plt.subplots(figsize=(16,8))
ax.hist(df['Production'], bins = 60, color='skyblue', edgecolor='black')
ax2 = ax.twinx()
df["Production"].plot(kind="kde", ax=ax2, color='orange', linewidth= 1.5)
ax.set_xlabel('Production (MWh)', fontweight='bold',fontsize=18)
ax.set_ylabel('Frequency', fontweight='bold',fontsize=18)
ax.set_title('Hourly Production Distribution', fontsize=20, fontweight='bold', pad=20)
ax2.set_ylabel("Density", fontweight='bold', fontsize=18)

#Adding a text box at the corner of the chart
stat_text = f"Mean: {df['Production'].mean():.0f} MWh \nMedian: {df['Production'].median():.0f} MWh \nStd Dev: {df['Production'].std():.0f} "
ax.text(
    0.98, 0.98, stat_text,
    fontsize = 12, fontweight='bold',
    transform=ax.transAxes,
    va = 'top',
    ha = 'right',
    bbox=dict(boxstyle='round', facecolor = 'wheat', alpha=0.7)
)

# using spines to thicken the border
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
for spine in ax2.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

# removing gridlines 
ax.grid(False)
ax2.grid(False)
fig.tight_layout()

# Saving image
os.path.exists('Kaggle_datasets/Hourly_Energy_Distribution.png') or fig.savefig("Hourly_Energy_Distribution.png",
            dpi = 600,
            facecolor = 'white',
            bbox_inches = 'tight')
print("Histogram Loaded")

# Energy source distributition through PIe chart
fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))

source_count = df['Source'].value_counts()
pie_colors = ["#1b4965", "#62b6cb"]
ax1.pie(
    source_count,
    labels=source_count.index,
    autopct='%1.1f%%',
    startangle=45,
    wedgeprops={'edgecolor':'black', 'linewidth': 2},
    colors=pie_colors,
    textprops = {'fontsize': 14, 'weight': 'bold', 'color': 'white'}
)

ax1.set_title("Distribution by Energy Source", fontsize=20, fontweight='bold')


source_produc = df.groupby('Source')['Production'].sum().sort_values(ascending=False)

bars = ax2.bar(source_produc.index, source_produc.values, color=pie_colors, edgecolor='black', linewidth=2)
ax2.set_title("Total Production by Energy Source", fontsize=20, fontweight='bold', pad = 20)
ax2.set_xlabel("Energy Source", fontsize = 18, fontweight='bold')
ax2.set_ylabel("Total Production (MWh)", fontsize = 18, fontweight='bold')
ax2.tick_params(axis='both', labelsize=14)

for bar in bars:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width()/2.,
        height,
        f"{height/1e6:.2f}M", va='bottom', ha='center', fontsize=12, fontweight='bold'       
    )

for spine in ax2.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
fig1.tight_layout()
os.path.exists('Kaggle_datasets/Overall_ Distribution_and_Production_chart.png') or fig1.savefig("Overall_ Distribution_and_Production_chart.png", dpi=800,bbox_inches='tight' )
print("Overall_ Distribution_and_Production_chart loaded")

fig, ax = plt.subplots(figsize=(16,8))

seasons_order = ['Winter', 'Summer', 'Spring', 'Fall']
seasonal_data = df[df['Season'].isin(seasons_order)]

ax.boxplot(
    [seasonal_data[seasonal_data['Season'] == s]['Production'].values for s in seasons_order],
    labels = seasons_order, patch_artist=True,
    boxprops=dict(linewidth=2, edgecolor='black'),
    whiskerprops=dict(linewidth=2, color='black'),
    capprops=dict(color='black', linewidth=2),
    medianprops=dict(color='red', linewidth=2),
    flierprops=dict(marker='o', markerfacecolor='coral', markersize=6, markeredgecolor='black', linewidth=1.5)
    )
ax.set_title('Energy Production by Season', fontsize=20, fontweight='bold',pad=20)
ax.set_xlabel('Season', fontsize = 16, fontweight='bold')
ax.set_ylabel('Production (MWh)', fontsize = 16, fontweight='bold')
ax.tick_params(axis='both', labelsize=14)

for i,season in enumerate(seasons_order, 1):
    mean_value = seasonal_data[seasonal_data['Season'] == season]["Production"].mean()
    ax.plot(i, mean_value,
            marker='D',
            markersize=12,
            color='green',
            markeredgecolor='black',
            markeredgewidth=1.5,
            label='Mean' if i ==1 else ""
            )

ax.legend(frameon=True, framealpha=1, prop={'weight':'bold'},edgecolor='black', fontsize='large')

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
print("Production by season chart loaded")
os.path.exists('Kaggle_datasets/Production_by_season.png') or fig.savefig("Production_by_season.png", dpi=600, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(14,8))

hourly_mean = df.groupby('Start_Hour')['Production'].agg(['mean', 'std'])

ax.plot(hourly_mean.index, hourly_mean['mean'],
        marker='o', markeredgecolor='black', markeredgewidth=1.5, markersize= 10,
        label='Mean Production', markerfacecolor='#5e548e' )
ax.fill_between(hourly_mean.index,
                hourly_mean['mean']- hourly_mean['std'],
                hourly_mean['mean']+ hourly_mean['std'],
                facecolor='#be95c4', alpha=0.4
                )

ax.legend()
ax.set_title('Hourly Production Patterns', fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel("Start Hour", fontsize=16, fontweight='bold')
ax.set_ylabel("Production (MWh)", fontsize=16, fontweight='bold')
ax.set_xticks(range(0, 24, 1))
ax.tick_params(labelsize=14, axis='both')
ax.grid(linewidth=0.2)

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.tight_layout()
print("Hourly production pattern chart loaded")
img = 'Hourly_Production_Pattern.png'
os.path.exists(f'Kaggle_datasets/{img}') or fig.savefig(img)

fig, ax = plt.subplots(figsize=(16,8))
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_avg = df.groupby('Month_Name')['Production'].mean().reindex(month_order)
colors = plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(monthly_avg)))
bars = ax.bar(range(len(monthly_avg)), monthly_avg.values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_title('Monthly Average Production', fontsize= 20, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=16, fontweight='bold')
ax.set_ylabel('Production (MWh)', fontsize=16, fontweight='bold')
ax.set_xticks(range(0,12,1))
ax.set_xticklabels([a[:3] for a in monthly_avg.index], fontsize=14, fontweight='bold')
ax.tick_params(axis='y', labelsize=14)
ax.grid(alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2.,height, f'{height:.0f}',
        ha='center', va='bottom', fontweight='bold'
    )

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
print("Monthly average produciton chart loaded")
img = "Monthly_average_production.png"
os.path.exists(f'Kaggle_datasets/{img}') or fig.savefig(img, dpi=600)


fig, ax = plt.subplots(figsize=(16,8))

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
days_avg = df.groupby('Day_Name')['Production'].agg(['mean','std']).reindex(days_order)
colors_day = plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(days_avg)))

bars = ax.bar(range(len(days_order)), days_avg['mean'].values,
              yerr = days_avg['std'].values,
              color=colors_day, edgecolor='black', linewidth=1.5,
              error_kw={'elinewidth': 1.5},
              capsize=10)

ax.set_title('Daily Avg Production', fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel('Day of Week', fontsize=16, fontweight='bold')
ax.set_ylabel('Production (MWh)', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(days_order)))
ax.set_xticklabels([a[:3] for a in days_avg.index], fontsize=14, fontweight='bold')
ax.tick_params(axis='y', labelsize=14)

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
print("Daywise avg prod chart loaded")
img = 'Daily avg production.png'
os.path.exists(f'Kaggle_datasets/{img}') or fig.savefig(img, dpi=600)

fig, ax = plt.subplots(figsize=(16,8))

Prod_source = ['Wind', 'Solar']
color_source= {'Wind':'#FFAAB8', 'Solar': '#A8DF8E'}

for source in Prod_source:
    avg_source = df[df['Source']==source].groupby('Start_Hour')['Production'].mean()
    ax.plot(avg_source.index, avg_source.values, linewidth=2.5, marker='o', markeredgecolor='black', markeredgewidth=1.5, label=source, color=color_source[source])

ax.set_title('AVG Hourly Production pattern by Source', fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel('Hours of Day', fontsize=14, fontweight='bold')
ax.set_ylabel('Production (MWh)', fontsize=14, fontweight='bold')
ax.set_xticks(range(0,24,2))
ax.tick_params(axis='both', labelsize=14)
ax.legend()

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
img='Avg_hr_prod_pattern.png'
print('AVG Hourly Production pattern by Source loaded')
plt.tight_layout()
os.path.exists(f'Kaggle_datasets/{img}') or fig.savefig(img, dpi=600)
