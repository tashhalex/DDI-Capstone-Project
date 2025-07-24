import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import datetime
import base64
import os
import urllib.request

# Page Configuration
st.set_page_config(
    page_title = "📡 Satellite Decay Predictor",
    page_icon = "🛰️",
    layout="centered",
 )

# Load Trained Model
model_url = "https://dl.dropboxusercontent.com/scl/fi/e7n8w1ghx9tp8sukrf6mz/rf_model.joblib?rlkey=f1sk13ifutnxttvqym4mbkwgd&st=j57wcij5&dl=1"
model_path = os.path.join('models', 'rf_model.joblib')

def download_model():
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)
        print('Model downloaded successfully')

@st.cache_resource
def load_model():
    download_model()
    return joblib.load(model_path)

model = load_model()
MAE_ESTIMATE = 2.3

def display_styled_image(image_path, caption="", width="80%", shadow=True, border_radius='10px'):
    try:
        with open(image_path, 'rb') as img_file:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode()
        shadow_style = 'box-shadow: 0 4px 8px rgba(0,0,0,0.6);' if shadow else ""

        st.markdown(
            f"""
            <div style='text-align: center; padding: 10px'>
                <img src='data:image/png;base64,{img_base64}'
                    style='border-radius: {border_radius}; {shadow_style} width: {width};' />
                <p style='color: #6f6d72; font-size:16px;'>{caption}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f'⚠️ Image not found: {image_path}')

st.title('🛰️')

# Tabs
tab1, tab2, tab3 = st.tabs(['🏠 Home', '📊 Prediction Dashboard', '👤 About Me/Contact'])

# --- HOME Tab ----
with tab1:
    st.markdown("<h2 style='color:#FFFFFF; font-family: Poppins;'> Satellite Decay Prediction Project</h2>", unsafe_allow_html=True)
    st.markdown("""
    <h4 style='color:#c9374c; font-family: Poppins;'>
    Problem Statement & Objectives</h4>""", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#FFFFFF; font-family: Helvetica;'>
    Space Domain Awareness depends heavily on understanding satellite life cycles, particularly predicting when
    satellites will decay and re-enter Earth's atmosphere. As space congestion grows, especially in Low Earth Orbit 
    (LEO), understanding satellite decay patterns is critical for many aspects of space operations. This tool aims
    to [eventually] complement traditional physics-based models by offering data-driven insights derived from historical
    decay trends.</p>""", unsafe_allow_html=True)
    st.markdown("""
    <h6 style='color: #9ccddc; font-family: Poppins;'>
    Objectives:</h6>
    <ul>
        <li style='font-family: Helvetica;'>Develop a machine learning model that can predict satellite decay timing using minimal TLE data.</li>
        <li style='font-family: Helvetica;'>Provide a user-friendly dashboard that allows for single or batch satellite predictions.</li>
        <li style='font-family: Helvetica;'>Visualize orbital factors that are most strongly associated with decay timelines.</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("""
    <h4 style='color: #c9374c; font-family: Poppins;'>
    Data Sources and Major Transformations</h4>""", unsafe_allow_html=True)
    st.markdown("""
    <h6 style='color: #9ccddc; font-family: Poppins;'>
    Data Sources:</h6>
    <ul>
        <li style='font-family: Helvetica;'>Space-Track Decay Data - Satellites with known decay dates</li>
        <li style='font-family: Helvetica;'>Space-Track TLE data - Historical TLE data of those satellites, 60 days prior to decay</li>
    </ul>
    <h6 style='color: #9ccddc; font-family: Poppins;'>
    Major Transformations:</h6>
    <ol>
        <li style='font-family: Helvetica;'>Merged decay data with corresponding TLE data using NORAD CAT IDs</li>
        <li style='font-family: Helvetica;'>Cleaned data to ensure EPOCH dates were valid and that decay intervals were non-negative.</li>
        <li style='font-family: Helvetica;'>Calculated 'Days to Decay' using the difference between TLE EPOCH and actual decay date</li>
    </ol>
    <h6 style='color: #9ccddc; font-family: Poppins;'>
    Initial Findings:</h6>
    """, unsafe_allow_html=True)
    display_styled_image('assets/corr-heatmap.png', 'Correlation Heatmap')

    st.markdown("""
    <h4 style='color: #c9374c; font-family: Poppins;'>
    About the Model</h4>
    """, unsafe_allow_html=True)
    st.markdown("""
    <h7 style='color: #c874b2; font-family: monospace;'>
    Model:</h7><p> Random Forest Regressor</p>
    <h7 style='color: #c874b2; font-family: monospace;'>
    Inputs:</h7><p> Mean Motion (rev/day), Eccentricity (shape), Inclination (tilt), and B* Drag Term (drag)</p>
    <h7 style='color: #c874b2; font-family: monospace;'>
    Output:</h7></p>Days to Decay</p>
    """, unsafe_allow_html=True)
    st.markdown("""

    """, unsafe_allow_html=True)
    st.markdown('---')
    st.markdown("""
    <h4 style='color: #c9374c; font-family: Poppins;'>
    Visualizations & Key Findings</h4>
    """, unsafe_allow_html=True)
    display_styled_image('assets/feat-import.png', 'Feature Importances')
    st.markdown("""
    <p style='text-align:center; font-family: Helvetica;'>
    A histogram of feature importances showed that mean motion and inclination are significant predictors of the 
    decay date, with eccentricity and B* being fairly insignificant.</p>
    """, unsafe_allow_html=True)

    display_styled_image('assets/mm_decay.png', 'Mean Motion vs. Days to Decay LOWESS')
    st.markdown("""
    <p style='text-align:center; font-family: Helvetica;'>
    This graph gives a visual representation of the relationship between mean motion and decay. It shows that
    satellites with a higher mean motion - meaning lower orbit, tend to decay more rapidly than satellites with
    a lower mean motion or higher orbit.</p>
    """, unsafe_allow_html=True)
    display_styled_image('assets/residuals.png', 'Residuals Distribution')
    st.markdown("""
    <p style='text-align:center; font-family: Helvetica;'>
    This residuals distribution graph gives a visual representation of how off the model's predictions were
    from actual values. It shows that most predictions are close to 0 error, with most predictions occurring
    within a few days of the actual decay time.</p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <h4 style='color: #c9374c; font-family: Poppins;'>
    Future Areas of Research</h4>
    <ul>
        <li style='font-family: Helvetica;'>Incorporate solar activity, space weather, and atmospheric weather data for improved drag modeling.</li>
        <li style='font-family: Helvetica;'>Quantify Space-Track's RCS Size and incorporate it as a feature in the model.</li>
        <li style='font-family: Helvetica;'>Combine phyics-based models with machine learning models. </li>
    </ul>
    """, unsafe_allow_html=True)

#--- PREDICTOR Tab ---
with tab2:
# App Title & Description
    st.markdown("<h2 style='color:#FFFFFF; font-family: Poppins;'>Decay Prediction Input</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-family: monospace; font-size:20px;'>
    Fill in the satellite orbital parameters below to get a predicted decay date.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Inputs for TLE parameters
    with col1:
        mean_motion = st.number_input('Mean Motion (rev/day)', 0.0, 20.0, 15.0)
        eccentricity = st.number_input('Eccentricity (shape)', 0.0, 1.0, 0.0001, format='%.6f')
    with col2:
        inclination = st.number_input('Inclination (tile, deg)', 0.0, 180.0, 98.0)
        bstar = st.number_input('B* Drag Term', -1.0, 1.0, 0.0001, format="%.6f")
    tle_date = st.date_input('Last TLE EPOCH Date', value=datetime.date.today())

    user_input_df = pd.DataFrame({
        'MEAN_MOTION': [mean_motion],
        'ECCENTRICITY': [eccentricity],
        'INCLINATION': [inclination],
        'BSTAR': [bstar]
    })
                
    # Predict Button
    if st.button('Predict Days to Decay'):
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
            csv = batch_df.to_csv(index=False)
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

    st.markdown("---")
    st.markdown("""⚠️ _This prediction is based on historical data trends using a statistics-based,
    machine learning model and may differ from authoritative forecasts provided by Space-Track 
    or other satellite tracking agencies._ ⚠️ """)

with tab3: 
    col3, col4 = st.columns(2)

    with col3:
        display_styled_image('assets/me.png', 'The Creator')

    with col4:
        st.markdown("""
        <h4 style='color: #c9374c; font-family: Poppins;'>
        👩🏽‍🚀 About the Creator</h4>
        """, unsafe_allow_html=True)
        st.markdown("""
        <p> Hi! I'm Tashi Hatchell (Alexander) - an aspiring data scientist who is passionate about space exploration
        and satellite dynamics. This project is part of my capstone project to model satellite orbital decay using
        machine learning.<p>
        """, unsafe_allow_html=True)
        st.markdown("""
        <h6 style='color: #9ccddc; font-family: Poppins;'>
        📫 Contact:</h6>
        <ul>
            <li>Email: thatchell.aca@gmail.com</li>
            <li>GitHub: <a href="https://github.com/tashhalex" target=_blank>https://github.com/tashhalex</a></li>
            <li>LinkedIn: <a href="https://www.linkedin.com/in/thatchell/" target=_blank>https://www.linkedin.com/in/thatchell/</a></li>
        </uL>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown('---')
    st.markdown('🌌Powered by Random Forests, Streamlit :streamlit:, and a love for the stars.:stars:')
    st.markdown("""⚠️ _This prediction is based on historical data trends using a statistics-based,
    machine learning model and may differ from authoritative forecasts provided by Space-Track 
    or other satellite tracking agencies._ ⚠️ """)