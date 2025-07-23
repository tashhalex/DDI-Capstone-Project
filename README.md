# Satellite Decay Predictiong Using a Machine Learning Model
DDI Final Capstone Project

### Problem Statement


### Project Objectives
- Develop a machine learning model that can predict satellite decay timing using minimal TLE data. 
- Determine what TLE features directly impact orbital decay rates. 

### Context
- With the increase of space congestion, particularly in Low Earth Orbit (LEO), understanding satellite decay patterns becomes critical for mission planning, satellite tracking, and deorbit operations. 

### Data Sources and Major Transformations
- Space-Track: decay data for satellites with known decay dates, and historical TLE data of those satellites collected within 60 days prior to decay.
- Major Transformations:
    1. Merged decay records with corresponding TLE data using NORAD catalog ID numbers
    2. Cleaned data to ensure epoch dates were valid and that decay intervals were non-negative.
    3. Derived 'days to decay' target variable based on the difference between TLE epoch and recorded decay date.

### Visualizations and Key Findings
Mean Motion vs Days to Decay
- ![alt text](<Mean Motion vs Days to Decay with LOWESS.png>)
- Strong inverse relationship, satellites with higher mean motion decay faster

Feature Importance Analysis
- ![alt text](<Feature Importances.png>)
- Mean Motion & Inclination are dominant predictors

Residual Analysis
- ![alt text](<Residuals Distributed.png>)
- Most predictions fell within +/- 2.3 days of actual decay dates. One extreme outlier was identified during evaluation. 


### Future Areas of Research
- Incorporate Solar Activity for improved drag modeling
- Combining physics-based models with machine learning models

### Conclusion 
