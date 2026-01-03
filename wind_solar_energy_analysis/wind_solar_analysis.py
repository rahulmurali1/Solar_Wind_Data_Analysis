import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
plt.tight_layout()

# Saving image
# plt.savefig("Hourly_Energy_Distribution.png",
#             dpi = 600,
#             facecolor = 'white',
#             bbox_inches = 'tight')
print("Histogram Loaded")
plt.show()