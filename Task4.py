import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df=pd.read_csv("C:\\Users\\dell\\Downloads\\US_Accidents_March23.csv")
print("Shape of data:", df.shape)

# Check missing values
print("\nTop 10 Columns with Most Missing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# Convert time columns to datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'].str[:19], format="%Y-%m-%d %H:%M:%S", errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

# Extract temporal features
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
df['Month'] = df['Start_Time'].dt.month
df['Year'] = df['Start_Time'].dt.year

# Handle missing weather data
weather_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median(numeric_only=True))

# Plot 1: Accidents by hour of day
plt.figure(figsize=(10, 5))
sns.countplot(x='Hour', data=df, palette='coolwarm')
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Accidents')
plt.tight_layout()
plt.show()

# Plot 2: Top 10 weather conditions during accidents
plt.figure(figsize=(12, 6))
top_weather = df['Weather_Condition'].value_counts().nlargest(10).index
sns.countplot(
    x='Weather_Condition',
    data=df[df['Weather_Condition'].isin(top_weather)],
    order=top_weather,
    palette='magma'
)
plt.title('Top 10 Weather Conditions During Accidents')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 3: Scatter plot of Temperature vs Visibility
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Temperature(F)', y='Visibility(mi)', data=df)
plt.title('Temperature vs Visibility')
plt.xlabel('Temperature (°F)')
plt.ylabel('Visibility (mi)')
plt.tight_layout()
plt.show()

# Plot 4: Severity vs Visibility
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Severity', y='Visibility(mi)', data=df)
plt.title('Severity vs Visibility')
plt.xlabel('Severity')
plt.ylabel('Visibility (mi)')
plt.tight_layout()
plt.show()

# Plot 5: Severity by Traffic Features
features = ['Traffic_Signal', 'Junction', 'Stop', 'Crossing']
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Severity', hue=feature, data=df, palette='pastel')
    plt.title(f'Severity by {feature}')
    plt.tight_layout()
    plt.show()

# Plot 6: Accident Hotspots Map for California
import folium
from folium.plugins import HeatMap

subset = df[df['State'] == 'CA'].dropna(subset=['Start_Lat', 'Start_Lng']).sample(5000, random_state=42)
lat_center = subset['Start_Lat'].mean()
lng_center = subset['Start_Lng'].mean()

accident_map = folium.Map(location=[lat_center, lng_center], zoom_start=6)
heat_data = [[row['Start_Lat'], row['Start_Lng']] for _, row in subset.iterrows()]
HeatMap(heat_data).add_to(accident_map)
accident_map.save("accident_hotspot_heatmap.html")

# Plot 7: Accidents by Day of the Week
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df['Day_Name'] = df['Day'].apply(lambda x: days[x])

plt.figure(figsize=(10, 5))
sns.countplot(x='Day_Name', data=df, order=days, palette='Set2')
plt.title('Accidents by Day of the Week')
plt.xlabel('Day')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Key Observations (summarized from your notes)
'''
- Most accidents occur during morning and evening rush hours.
- Weather conditions like fog, rain, and snow increase accident likelihood.
- Junctions, crossings, and traffic signals are common accident points.
- Weekdays have higher accident rates than weekends.
- Severity is slightly higher when visibility is low.
'''


