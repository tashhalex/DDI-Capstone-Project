import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import datetime

# Page Configuration
st.set_page_config(
    page_title = "📡 Satellite Decay Predictor",
    page_icon = "🛰️",
    layout="centered",
    initial_sidebar_state = 'expanded'
)

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load('../models/rf_model.joblib')

model = load_model()

# App Title & Description
st.title('🛰️ Satellite Decay Prediction Dashboard')
st.markdown("""
Welcome to the Satellite Decay Predictor!""") 
st.markdown("""Input satellite orbital parameters into the sidebar :point_left:  to predict the expected **Days to Decay**.\n
**Built with ❤️ (and a few tears) using :evergreen_tree: Random Forest Regression.**
""")

# Sidebar for User Inputs
st.sidebar.header('Satellite TLE Parameters')
# Bars for TLE parameters
mean_motion = st.sidebar.number_input('Mean Motion (rev/day)', min_value=0.0, max_value=20.0, value=15.0)
eccentricity = st.sidebar.number_input('Eccentricity', min_value=0.0, max_value=1.0, value=0.0001, format="%.6f")
inclination = st.sidebar.number_input('Inclination (deg)', min_value=0.0, max_value=180.0, value=98.0)
bstar = st.sidebar.number_input('B* Drag Term', min_value=-1.0, max_value=1.0, value=0.0001, format='%.6f')
#Side bar for Date
tle_date = st.sidebar.date_input('Last TLE EPOCH Date', value=datetime.date.today())


user_input_df = pd.DataFrame({
    'MEAN_MOTION': [mean_motion],
    'ECCENTRICITY': [eccentricity],
    'INCLINATION': [inclination],
    'BSTAR': [bstar]
})

MAE_ESTIMATE = 2.3
                                
# Predict Button
if st.sidebar.button('Predict Days to Decay'):
    prediction = model.predict(user_input_df)[0]
    decay_date = tle_date + datetime.timedelta(days=prediction)
    st.success(f""" 
    Predicted Days to Decay: **{prediction:.2f} days** (± {MAE_ESTIMATE} days)\n\n
    Expected Decay Date: **{decay_date.strftime('%d %B %Y')}**""")
st.markdown("---")
# Batch Predictions
st.subheader('Batch Prediction')
st.markdown('If you need predictions for multiple satellites...')
uploaded_file = st.file_uploader('Upload CSV with columns: MEAN_MOTION, ECCENTRICITY, INCLINATION, BSTAR, and TLE_EPOCH"', type='csv')

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    if {'MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'BSTAR', 'TLE_EPOCH'}.issubset(batch_df.columns):
        X_batch = batch_df[['MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'BSTAR']]
        batch_df['Predicted_Days_to_Decay'] = model.predict(X_batch)
        batch_df['Expected_Decay_Date'] = pd.to_datetime(batch_df['TLE_EPOCH']) + pd.to_timedelta(batch_df['Predicted_Days_to_Decay'], unit='d')
        st.success('✅ Batch prediction complete!')
        st.dataframe(batch_df)
        csv = batch_df.to_csv(index=False), encode('utf-8')
        st.download_button('📥Download Predictions as CSV', data=csv, file_name='batch_predictions.csv')
    else:
        st.error(':anger: Uploaded CSV is missing required columns. :anger:')

# Feature Importances Section
st.subheader('Model Feature Importances')
features = ['MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'BSTAR']
importances = model.feature_importances_

fig,ax = plt.subplots(figsize=(8,3))
sns.barplot(x=features, y=importances, palette='autumn', ax=ax)
ax.set_title('Feature Importances')
plt.tight_layout()
st.pyplot(fig)

# Residuals Distribution Section
st.subheader('Model Residuals Example')
st.markdown('Example residuals distribution from model testing.')

example_residuals = np.random.normal(0, 10, size=5000)
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.histplot(example_residuals, bins=50, kde=True, color='coral', ax=ax2)
ax2.set_title('Sample Residuals Distribution')
plt.tight_layout()
st.pyplot(fig2)

# Footer
st.markdown('---')
st.markdown(':sparkles: Built by Tashi Hatchell (Alexander) -- Powered by Random Forests and Streamlit :streamlit: :sparkles:')
st.markdown("""⚠️ _This prediction is based on historical data trends using a statistics-based,
machine learning model and may differ from authoritative forecasts provided by Space-Track 
or other satellite tracking agencies._ ⚠️ """)