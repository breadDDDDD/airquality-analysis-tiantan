import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#geolocation
geo_location = pd.DataFrame({
    "lat" : [39.8837],
    "lon" : [116.4121],
    "name" : "Tiantan"
})

df = pd.read_csv("PRSA_Data_Tiantan_20130301-20170228.csv")
df = df.fillna(method='ffill')
dates = ["year", "month", "day", "hour"]
order = [ "datetime", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]

df["datetime"] = pd.to_datetime(df[dates])
df = df[order]
df.set_index('datetime', inplace=True)

#short description
description = df.describe(include="all")

#monthly resample
df_monthly = df.resample('M').mean()

#Bagian Streamlit
max_width = 700

#Header
st.title("Air Quality di Tiantan (2013 - 2017)")
st.subheader("""
            **MC-50**
           """)

#Body
st.header("Background")
st.write("Air quality is primarily influenced by pollutants like PM2.5, PM10, NO₂, SO₂, CO, and O₃. PM2.5 (fine particulate matter) consists of microscopic particles that can penetrate deep into the lungs, posing serious health risks. PM10 includes slightly larger particles that can still cause respiratory issues. High levels of these pollutants can lead to air pollution, causing smog, reduced visibility, and health concerns like asthma and lung disease.")
st.write("Tiantan Station, located near the Temple of Heaven in Beijing, China, is a key air quality monitoring site. It provides real-time data on various pollutants, helping assess the city's air pollution levels and trends. This station is crucial for tracking changes in air quality and guiding public health recommendations.")

map = px.scatter_mapbox(
    geo_location, lat="lat", lon="lon", text="name",
    zoom=12, mapbox_style="carto-positron",
    size=[15],  
    color_discrete_sequence=["red"] 
)
st.plotly_chart(map)

st.subheader("Business Questions")
st.write("""
       - Is there a connection between Pollutant PM10 and temperature in Tiantan city?
       - How is the trend of each type of Pollutant in 2013-2017 in Tiantan city?
         """)

st.header("Overview")

st.subheader("Libraries Used in this Analysis")
st.code("""
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        import seaborn as sns
        from sklearn.preprocessing import MinMaxScaler
        """, language="python")

st.subheader("Dataset")
st.write("The data are air quality measurements for pollutants and environmental factors in Tiantan in 2013-2017 which have been cleaned.")
df_csv =  df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data= df_csv,
    file_name="AirQualityTiantan2013-2017.csv",
    mime="text/csv",
)
st.dataframe(data = df, width = max_width, height=500)
df_csv =  df.to_csv(index=False).encode("utf-8")

st.header("Statistics")
st.write("Here are is a statistical summary of the dataset")
st.dataframe(data = description, width = max_width, height = 200)
st.write("""
        - Data range for each type of pollutant:
        - PM2.5: 3 - 821, mean: 82.25
        - PM10: 2 - 988, mean: 106.65
        - SO2: 0.57 - 273, mean: 14.47
        - NO2: 2 - 241, mean: 53.27
        - CO: 100 - 10000, mean: 1308.27

        - For temperature: -16.80 - 41.10, mean: 13.66
        - It can be noted that the range of pollutants is quite varied
         """)

st.subheader("Correlation Matrix")
corr_matrix = df.corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix,cmap='coolwarm', annot=True, fmt=".2f")
st.pyplot(fig)

st.write("""
        - It can be seen that the types of pollutants PM2.5, PM10, SO2, NO2, and CO have a correlation of 0.38 - 0.89. This means that these features have a fairly good correlation and are positive

        - Meanwhile, temperature has a negative correlation with PM2.5, PM10, SO2, NO2, and CO. This means that the higher the value of the pollutant, the smaller the temperature value and vice versa
        """)

st.header("Explanatory Analysis")
st.subheader("Is there a connection between PM10 Pollutant and temperature in Tiantan city?")
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(10, 10))

    line1 = ax1.plot(df_monthly.index, df_monthly["PM10"], label='PM10', linewidth=2, color='blue', marker='o')
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Konsentrasi PM10 (µg/m³)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    lineTemp = ax2.plot(df_monthly.index, df_monthly["TEMP"], label='Temperature', linewidth=2, color='red', marker='o')
    ax2.set_ylabel('Temperature (°C)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')


    plt.title('Trends Pollutan PM10 dan Temperature pada Tahun 2013-2017', fontsize=14, pad=14)
    lines_total = line1 + lineTemp
    labels = [l.get_label() for l in lines_total]

    ax1.legend(lines_total, labels, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

    fig1.tight_layout()
    st.pyplot(fig1)
    
with col2:
    fig2, ax = plt.subplots()
    corr_matrix = df[["TEMP", "PM10"]].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f")
    st.pyplot(fig2)

st.write("""
         The correlation between temperature and PM10 was found to be weak, indicating that temperature alone does not have a strong influence on PM10 concentrations. This indicates that other factors, such as emissions from industrial activities, vehicle traffic, and seasonal variations in warming, play a more significant role in PM10 levels than temperature changes.
         """)

st.subheader("What is the trend of each type of Pollutant in 2013-2017 in Tiantan city?")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["PM2.5", "PM10", "SO2", "NO2", "CO"])

def plot_pollutant(pollutant):
  fig,ax = plt.subplots()
  ax.plot(df_monthly.index, df_monthly[pollutant],
          label=pollutant,
          linewidth=1.5,
          color='red',
          markersize=6,
          marker='o',
          markerfacecolor='white',
          markeredgewidth=1.5,
          markeredgecolor='royalblue')

  ax.set_title(f'Average Pollutant {pollutant} in Tiantan', fontsize=14, pad=14)
  ax.set_xlabel('Period (Month-Year)', fontsize=12, labelpad=12)
  ax.set_ylabel(f'Consentration {pollutant} (µg/m³)', fontsize=12, labelpad=12)
  ax.set_ylim(0, df_monthly[pollutant].max() * 1.1)
  ax.legend(loc='upper right', frameon=False, fontsize=12)
  ax.grid(True, which='both', linestyle='--', linewidth=0.5)
  st.pyplot(fig)

with tab1:
    st.subheader("PM2.5")
    plot_pollutant("PM2.5")
    st.write("""
        **Description**
        - It can be noted that there was a drastic increase in January 2016
        - There were peaks in mid-2014, January 2016, and January 2017
        - In early-mid 2016 there was a fairly drastic decrease
             """)

with tab2:
    st.subheader("PM10")
    plot_pollutant("PM10")
    st.write("""
            **Description**
            - Trend is quite similar to PM2.5
            - Peaks in mid-2014, January 2016, and January 2017 as well
                        """)

with tab3:
    st.subheader("SO2")
    plot_pollutant("SO2")
    st.write("""
            **Description**
            - Experienced an overall decline
            - The peak was in mid-2014
                        """)
with tab4:
    st.subheader("NO2")
    plot_pollutant("NO2")
    st.write("""
            **Description**
            - There were peaks in mid-2014, January 2016, and January 2017 as well
                        """)
    
with tab5:
    st.subheader("CO")
    plot_pollutant("CO")
    st.write("""
            **Description**
            - Has very high concentration
            - Also has peaks in mid-2014, January 2016, and January 2017
                        """)
    
    
st.write("**Combined**")

fig, ax = plt.subplots()
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_monthly[["PM2.5", "PM10", "SO2", "NO2", "CO"]])
df_scaled = pd.DataFrame(df_scaled, columns=["PM2.5", "PM10", "SO2", "NO2", "CO"])

ax.plot(df_monthly.index, df_scaled["PM2.5"], label = "PM2.5")
ax.plot(df_monthly.index, df_scaled["PM10"], label ="PM10")
ax.plot(df_monthly.index, df_scaled["SO2"], label ="SO2")
ax.plot(df_monthly.index, df_scaled["NO2"], label ="NO2")
ax.plot(df_monthly.index, df_scaled["CO"], label = "CO")

ax.set_title(f'Pollutant Trends over Time Period', fontsize=14, pad=14)
ax.set_xlabel('Period (Month-Year)', fontsize=12, labelpad=12)
ax.set_ylabel(f'Trend Pollutant', fontsize=12, labelpad=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()
st.pyplot(fig)
st.write("""
         Although each type of pollutant has a different pattern of increase and decrease, there are similarities in the peak data in the same period, namely between 01-2014 to 07-2014. Although the trends of several types of pollutants are similar, the amount of pollutants detected in µg/m³ varies, with CO pollutants having the highest concentration.
         """)

#Footer
st.markdown("---")
st.markdown(
    '[![GitHub](https://img.shields.io/badge/GitHub-breadDDDDD-blue?logo=github)](https://github.com/breadDDDDD)'
)