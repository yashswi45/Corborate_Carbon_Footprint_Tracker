import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('PNQ_AQI_with_CarbonFootprint.csv', parse_dates=['Date'], dayfirst=True)
df = df.sort_values('Date')

# Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayOfYear'] = df['Date'].dt.dayofyear

# Rolling averages for AQI trends
df['AQI_7day_avg'] = df['AQI'].rolling(window=7).mean()
df['AQI_14day_avg'] = df['AQI'].rolling(window=14).mean()
df = df.dropna()

# List of corporate locations in Pune
locations = ["Hinjewadi", "Kharadi", "Magarpatta", "Baner", "Viman Nagar", "Koregaon Park"]

# Train models separately for each location
models = {}
features = ['SO2', 'NOx', 'RSPM', 'SPM', 'Carbon_Footprint', 'PM2.5',
            'Cigarettes_per_Day', 'Year', 'Month', 'Day', 'DayOfWeek',
            'DayOfYear', 'AQI_7day_avg', 'AQI_14day_avg']

for location in locations:
    location_df = df[df['Location'] == location]

    if location_df.shape[0] < 50:  # Avoid training on very small datasets
        print(f"Skipping {location} due to insufficient data")
        continue

    # Train-test split (last 30 days as test)
    test_size = 30
    train = location_df.iloc[:-test_size]
    test = location_df.iloc[-test_size:]

    X_train = train[features]
    y_train = train['AQI']

    # Train XGBoost model
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model for this location
    model_filename = f'xgboost_aqi_{location}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

    models[location] = model

print("✅ Models trained and saved for all locations.")

# Function to load model for a given location
def load_model(location):
    model_filename = f'xgboost_aqi_{location}.pkl'
    with open(model_filename, 'rb') as f:
        return pickle.load(f)

# Function to classify AQI into zones
def classify_aqi_zone(aqi_value):
    if aqi_value > 102:
        return "Red"
    elif 102 <= aqi_value <= 102:
        return "Yellow"
    else:
        return "Green"

# Function to predict AQI for a given location and start date
def predict_aqi(start_date, location):
    if location not in locations:
        print("❌ Invalid location. Choose from:", locations)
        return None

    try:
        model = load_model(location)
    except FileNotFoundError:
        print(f"❌ No trained model available for {location}.")
        return None

    start_date = pd.to_datetime(start_date)
    future_dates = pd.date_range(start=start_date, periods=7)

    # Get last known values for the location
    location_df = df[df['Location'] == location]

    future_data = pd.DataFrame({
        'Date': future_dates,
        'Location': location,
        'SO2': [location_df['SO2'].mean()] * 7,
        'NOx': [location_df['NOx'].mean()] * 7,
        'RSPM': [location_df['RSPM'].mean()] * 7,
        'SPM': [location_df['SPM'].mean()] * 7,
        'Carbon_Footprint': [location_df['Carbon_Footprint'].mean()] * 7,
        'PM2.5': [location_df['PM2.5'].mean()] * 7,
        'Cigarettes_per_Day': [location_df['Cigarettes_per_Day'].mean()] * 7,
        'Year': future_dates.year,
        'Month': future_dates.month,
        'Day': future_dates.day,
        'DayOfWeek': future_dates.dayofweek,
        'DayOfYear': future_dates.dayofyear,
        'AQI_7day_avg': [location_df['AQI_7day_avg'].iloc[-1]] * 7,
        'AQI_14day_avg': [location_df['AQI_14day_avg'].iloc[-1]] * 7,
    })

    X_future = future_data[features]
    future_predictions = model.predict(X_future)

    future_data['Predicted_AQI'] = future_predictions
    future_data['AQI_Zone'] = future_data['Predicted_AQI'].apply(classify_aqi_zone)
    future_data['Cigarettes_per_Day'] = location_df['Cigarettes_per_Day'].mean()

    return future_data

# User Input
start_date = input("Enter the start date (YYYY-MM-DD) for AQI prediction: ")
location = input(f"Enter the location ({', '.join(locations)}): ")

# Predict and Append to CSV
predictions_df = predict_aqi(start_date, location)
if predictions_df is not None:
    print(predictions_df)

    # Save predictions to a new CSV file
    predictions_df.to_csv('AQI_Predicted.csv', mode='a', index=False,
                          header=not pd.io.common.file_exists('AQI_Predicted.csv'))
    print("✅ Predictions appended to AQI_Predicted.csv")

    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df[df['Location'] == location]['Date'], df[df['Location'] == location]['AQI'], label='Historical AQI')
    plt.plot(predictions_df['Date'], predictions_df['Predicted_AQI'], 'r--', label='Predicted AQI')

    # Color-coded zones
    for i in range(len(predictions_df)):
        color = 'red' if predictions_df.iloc[i]['AQI_Zone'].startswith("Red") else \
                'yellow' if predictions_df.iloc[i]['AQI_Zone'].startswith("Yellow") else 'green'
        plt.scatter(predictions_df.iloc[i]['Date'], predictions_df.iloc[i]['Predicted_AQI'], color=color, s=80)

    plt.legend()
    plt.title(f'AQI Prediction & Zones for {location}')
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.xticks(rotation=45)
    plt.show()



