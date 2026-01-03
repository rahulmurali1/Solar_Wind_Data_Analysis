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
fig1.tight_layout
os.path.exists('Kaggle_datasets/Overall_ Distribution_and_Production_chart.png') or fig1.savefig("Overall_ Distribution_and_Production_chart.png", dpi=800,bbox_inches='tight' )
print("Overall_ Distribution_and_Production_chart loaded")
plt.show()
