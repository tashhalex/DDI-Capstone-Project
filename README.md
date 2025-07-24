# Satellite Decay Predictiong Using a Machine Learning Model
DDI Final Capstone Project

### Problem Statement


### Project Objectives
- Develop a machine learning model that can predict satellite decay timing using minimal TLE data. 
- Determine what 2-Line Element (TLE) features directly impact orbital decay rates. 

### Context
- With the increase of space congestion, particularly in Low Earth Orbit (LEO), understanding satellite decay patterns becomes critical for mission planning, satellite tracking, and deorbit operations. 

### Data Sources and Major Transformations
- Space-Track: decay data for satellites with known decay dates, and historical TLE data of those satellites collected within 60 days prior to decay.
- Major Transformations:
    1. Merged decay records with corresponding TLE data using NORAD catalog ID numbers
    2. Cleaned data to ensure epoch dates were valid and that decay intervals were non-negative.
    3. Derived 'days to decay' target variable based on the difference between TLE epoch and recorded decay date.
    

![alt text](<Correlation Heatmap.png>)

### Model
- Random Forest Regressor 
    - Why? 
        - This model is like asking a team of experts to make a decision and each expert looks at the orbital data in a different way, they vote, and we average the answers. 
        - This type of model is better, in this case, than linear regression because there weren't clear linear relationships between the features. A linear regression model is similar to asking one expert who oversimplifies things and assumes are relationships are black and white. 
        - Satellite behavior is complex so we need a model that handles complex data well. 
- Features: Mean Motion (rev/day), Eccentricity (shape), Inclination (tilt) , BStar (drag), EPOCH Date
- Tried an XGBoost Model but it was not successful. 

### Visualizations and Key Findings
Mean Motion vs Days to Decay
![alt text](<Mean Motion vs Days to Decay with LOWESS.png>)
- Strong inverse relationship, satellites with higher mean motion decay faster

Feature Importance Analysis
![alt text](<Feature Importances.png>)
- Mean Motion & Inclination are dominant predictors, I thought it was interesting that B* Drag wasn't a huge factor because it's a way of modeling air resistance (drag) against a satellite. 

Residual Analysis
![alt text](<Residuals Distributed.png>)
- Most predictions fell within +/- 2.3 days of actual decay dates. One extreme outlier was identified during evaluation. 


### Future Areas of Research
- Incorporate Solar Activity for improved drag modeling
- Combining physics-based models with machine learning models

### Conclusion 
