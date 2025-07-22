import pandas as pd
import numpy as np
import seaborn as sns
import glob
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Data Cleaning & Merging Functions

# This function combines all of the separate TLE files into a dataframe then concats them into one large DF. 
    # Any duplicates are dropped and the EPOCH column is converted from an object to a datetime64 dtype.
def combine_tle_files(tle_path='data/raw_tles/*.csv'):
    file_list = glob.glob(tle_path)
    df_list = [pd.read_csv(file) for file in file_list]
    all_tle_df = pd.concat(df_list, ignore_index=True)
    all_tle_df.drop_duplicates(inplace=True)
    all_tle_df['EPOCH'] = pd.to_datetime(all_tle_df['EPOCH'], errors='coerce')
    all_tle_df.dropna(subset=['EPOCH'], inplace=True)
    all_tle_df.to_csv('data/all_tle_merged.csv', index=False)
    return all_tle_df

# This function loads and cleans the decay data csv by converting it into a DF, convertin EPOCH to datetime,
    # and dropping any null values. 
def clean_decay_data(decay_path='data/decay_data.csv'):
    decay_df = pd.read_csv(decay_path)[['NORAD_CAT_ID', 'DECAY_EPOCH']]
    decay_df['DECAY_EPOCH'] = pd.to_datetime(decay_df['DECAY_EPOCH'], errors='coerce')
    decay_df.dropna(subset=['DECAY_EPOCH'], inplace=True)
    return decay_df

# This function merges all_tle_df with decay_df, isolates the features needed for the model, then saves
    # those features as a separate dataframe
def merge_datasets(all_tle_df, decay_df):
    merged_df = all_tle_df.merge(decay_df, on='NORAD_CAT_ID', how='inner')
    merged_df['days_to_decay'] = (merged_df['DECAY_EPOCH'] - merged_df['EPOCH']).dt.days
    merged_df = merged_df[merged_df['days_to_decay'] >= 0]
    features_df = merged_df[['NORAD_CAT_ID', 'EPOCH', 'MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'BSTAR', 'days_to_decay']]
    features_df.to_csv('data/features.csv', index=False)
    return features_df

# EDA Functions
def perform_eda(features_df):
    print('\n Data Preview:')
    print(features_df.head())

    print('\n Data info:')
    print(features_df.info())

    print('\n Summary Statistics:')
    print(features_df.describe())

    print('\n Missing/Null Values:')
    print(features_df.isnull().sum())

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(features_df.corr(), annot=True, cmap='plasma', ax=ax)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

    for feature in ['MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'BSTAR', 'days_to_decay']:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(features_df[feature], bins=50, kde=True, ax=ax)
        ax.set_title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.show()
    
    for feature in ['MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'BSTAR', 'days_to_decay']:
        fig, ax = plt.subplots(figsize=(8,2))
        sns.boxplot(x=features_df[feature], ax=ax)
        ax.set_title(f'Boxplot of {feature}')
        plt.tight_layout()
        plt.show()
    
    for feature in ['MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'BSTAR']:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(x=features_df[feature], y=features_df['days_to_decay'], ax=ax, alpha=0.5)
        ax.set_title(f'{feature} vs. Days to Decay')
        plt.tight_layout()
        plt.show()
    
    filtered_df = features_df[features_df['days_to_decay'] < 5000].sample(5000, random_state=42)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x='MEAN_MOTION', y='days_to_decay', data=filtered_df, ax=ax, alpha=0.3, label='Sampled Data')
    z = sm.nonparametric.lowess(filtered_df['days_to_decay'], filtered_df['MEAN_MOTION'], frac=0.1)
    ax.plot(z[:,0], z[:, 1], color='red', linewidth=2, label='LOWESS Trend')
    ax.annotate('High Decay Zone', xy=(15,100), xytext=(12,1000),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
    ax.annotate('Stable Orbit Zone', xy=(5, 3000), xytext=(5, 4000),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
    ax.set_title('Mean Motion vs. Days to Decay with LOWESS Trend & Annotations')
    ax.legend()
    plt.tight_layout()
    plt.show()


# Modelin & Evaluation Functions
def train_eval_model(features_df):
    X = features_df[['MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'BSTAR']]
    y = features_df['days_to_decay']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Random Forest Test Set MAE: {mae:.2f} days')
    print(f'Random Forest Test Set R2: {r2:.2f}')

    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    cv_mae_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    print(f'Cross-Validated MAE per fold: {-cv_mae_scores}')
    print(f'Average Cross-Validated MAE: {-cv_mae_scores.mean():.2f} days')

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=['MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'BSTAR'], y=rf_model.feature_importances_, ax=ax, color='green', edgecolor='black')
    ax.set_title('Random Forest Feature Importances')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(y_test, y_pred, alpha=0.3, color='black')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', color='white')
    ax.set_xlabel('Actual Days to Decay')
    ax.set_ylabel('Predicted Days to Decay')
    ax.set_title('Actual vs. Predicted Days to Decay')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(residuals[(residuals > -50) & (residuals < 50)], bins=100, kde=True, ax=ax, color='green', alpha=0.7)
    ax.set_title('Residuals Distribution (Zoomed -50 to 50)')
    ax.set_xlabel('Prediction Error (Residuals)')
    plt.tight_layout()
    plt.show()

    joblib.dump(rf_model, 'models/rf_model.joblib')

if __name__ == '__main__':
    tle_df = combine_tle_files()
    decay_df = clean_decay_data()
    features_df = merge_datasets(tle_df, decay_df)
    perform_eda(features_df)
    train_eval_model(features_df)
    print('Full Workflow Complete')
