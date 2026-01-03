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
print(f"Shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

df = df[df['Source'] != "Mixed"]
print(f"filtered shape: {df.shape}")
print(f"Unique source: {df["Source"].unique()}")