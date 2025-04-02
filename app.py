import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings('ignore')

# 📌 Pune Locations
locations = {
    "Hinjewadi": [18.5912, 73.7380],
    "Kharadi": [18.5514, 73.9412],
    "Magarpatta": [18.5167, 73.9294],
    "Baner": [18.5646, 73.7769],
    "Viman Nagar": [18.5679, 73.9143],
    "Koregaon Park": [18.5362, 73.8937]
}

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #808080;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #696969;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .metric-title {
        color: #333333;
        font-size: 16px;
    }
    .metric-value {
        color: #333333;
        font-size: 24px;
        font-weight: bold;
    }
    .aqi-card {
        height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 5px;
    }
    .info-box {
        background-color: #838996;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .tab-content {
        padding: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Email function
def send_alert_email(receiver_email, subject, message, smtp_server, smtp_port, sender_email, sender_password):
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        # Add message body
        msg.attach(MIMEText(message, 'plain'))

        # Create server connection using context manager
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('PNQ_AQI_with_CarbonFootprint.csv', parse_dates=['Date'], dayfirst=True)
    df = df.sort_values('Date')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['AQI_7day_avg'] = df.groupby('Location')['AQI'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['AQI_14day_avg'] = df.groupby('Location')['AQI'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    return df

df = load_data()

def estimate_carbon_footprint(aqi_value):
    base_cf = 5.0
    return base_cf + (aqi_value * 0.05)

def classify_aqi_zone(aqi_value):
    if aqi_value > 150:
        return "Red"
    elif 120 <= aqi_value <= 150:
        return "Yellow"
    else:
        return "Green"

def load_model(location):
    model_filename = f'xgboost_aqi_{location}.pkl'
    try:
        with open(model_filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def predict_aqi(start_date, location):
    if location not in locations: return None
    model = load_model(location)
    if model is None: return None

    start_date = pd.to_datetime(start_date)

    future_dates = pd.date_range(start=start_date, periods=7)
    location_df = df[df['Location'] == location]
    if location_df.empty: return None

    last_values = location_df.iloc[-1]
    future_data = pd.DataFrame({
        'Date': future_dates,
        'Location': location,
        'SO2': [last_values['SO2']] * 7,
        'NOx': [last_values['NOx']] * 7,
        'RSPM': [last_values['RSPM']] * 7,
        'SPM': [last_values['SPM']] * 7,
        'Carbon_Footprint': [last_values['Carbon_Footprint']] * 7,
        'PM2.5': [last_values['PM2.5']] * 7,
        'Cigarettes_per_Day': [last_values['Cigarettes_per_Day']] * 7,
        'Year': future_dates.year,
        'Month': future_dates.month,
        'Day': future_dates.day,
        'DayOfWeek': future_dates.dayofweek,
        'DayOfYear': future_dates.dayofyear,
        'AQI_7day_avg': [last_values['AQI_7day_avg']] * 7,
        'AQI_14day_avg': [last_values['AQI_14day_avg']] * 7,
    })

    features = ['SO2', 'NOx', 'RSPM', 'SPM', 'Carbon_Footprint', 'PM2.5',
                'Cigarettes_per_Day', 'Year', 'Month', 'Day', 'DayOfWeek',
                'DayOfYear', 'AQI_7day_avg', 'AQI_14day_avg']

    X_future = future_data[features]
    future_predictions = model.predict(X_future)

    future_data['Predicted_AQI'] = future_predictions
    future_data['Predicted_Carbon_Footprint'] = future_data['Predicted_AQI'].apply(estimate_carbon_footprint)
    future_data['AQI_Zone'] = future_data['Predicted_AQI'].apply(classify_aqi_zone)

    return future_data

# Streamlit UI
st.title("🌍 Pune AQI Forecasting & Carbon Footprint Tracker")

# Navigation
tab1, tab2 = st.tabs(["AQI Forecast", "Email Alerts"])

with tab1:
    # Home Page Content
    if 'predictions_df' not in st.session_state:
        st.write("""
        This tool predicts Air Quality Index (AQI) and estimates carbon footprint for locations across Pune.

        ### About AQI and Carbon Footprint
        - **AQI (Air Quality Index)** measures how polluted the air is
        - Higher AQI values indicate greater air pollution and health concerns
        - Carbon footprint estimates the CO₂ emissions associated with these pollution levels
        - Industrial areas typically show higher AQI and carbon footprint values

        ### Cigarettes and Air Quality
        - The 'Cigarettes per Day' metric represents equivalent tobacco exposure from pollution
        - Poor air quality (high AQI) can be as harmful as smoking multiple cigarettes daily
        - Our model considers this relationship when making predictions

        To get started:
        1. Select a prediction date
        2. Choose a location
        3. Click the "Predict AQI" button
        """)

        st.markdown("### 🌐 Pune Monitoring Locations")
        m = folium.Map(location=[18.5204, 73.8567], zoom_start=12)
        for loc, coords in locations.items():
            folium.Marker(coords, popup=f"<b>{loc}</b>", tooltip=loc,
                          icon=folium.Icon(color='blue', icon='cloud')).add_to(m)
        folium_static(m, width=800)

    # Prediction Interface
    with st.sidebar:
        st.header("🔧 Prediction Parameters")
        start_date = st.date_input("📅 Select start date:", key="prediction_date_input")
        location = st.selectbox("📍 Select Location:",
                              list(locations.keys()),
                              key="location_selectbox")
        if st.button("Predict AQI", key="predict_button"):
            st.session_state.predictions_df = predict_aqi(start_date, location)

    # Display Results
    if 'predictions_df' in st.session_state and st.session_state.predictions_df is not None:
        predictions_df = st.session_state.predictions_df
        st.success(f"✅ AQI forecast generated for {location}")

        # AQI Forecast Cards
        st.markdown("### 📆 Next 7 Days AQI Forecast")
        cols = st.columns(7)
        color_map = {"Red": "#FF5733", "Yellow": "#FFC300", "Green": "#28A745"}

        for i, row in predictions_df.iterrows():
            with cols[i]:
                st.markdown(
                    f"""<div class="aqi-card" style='background-color: {color_map[row["AQI_Zone"]]}'>
                        <div style='font-weight:bold;'>{row['Date'].strftime('%a\n%d %b')}</div>
                        <div style='font-size:18px; margin:5px 0;'>{row['Predicted_AQI']:.0f}</div>
                        <div style='font-size:12px;'>{row['AQI_Zone']} Zone</div>
                    </div>""",
                    unsafe_allow_html=True
                )

        # AQI Trend with Zone Coloring
        st.markdown("### 📈 AQI Forecast Trend with Zones")
        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot with zone colors
        for i in range(len(predictions_df)):
            zone = predictions_df.iloc[i]['AQI_Zone']
            color = color_map[zone]
            ax.bar(predictions_df.iloc[i]['Date'], predictions_df.iloc[i]['Predicted_AQI'],
                   color=color, width=0.8, alpha=0.6)

        # Add line plot on top
        sns.lineplot(data=predictions_df, x='Date', y='Predicted_AQI',
                     color='black', linewidth=1.5, ax=ax, marker='o', markersize=8)

        # Customize plot
        ax.set_ylim(0, predictions_df['Predicted_AQI'].max() + 20)
        ax.set_ylabel('AQI Value', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

        # Create legend
        legend_elements = [Patch(facecolor=color_map['Green'], label='Good (AQI < 120)'),
                           Patch(facecolor=color_map['Yellow'], label='Moderate (120-150)'),
                           Patch(facecolor=color_map['Red'], label='Poor (AQI > 150)')]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Carbon Footprint Section
        st.markdown("### 🌱 Carbon Footprint Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Average Daily</div>
                <div class="metric-value">{:.2f} kg CO₂</div>
            </div>
            """.format(predictions_df['Predicted_Carbon_Footprint'].mean()), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Total 7-Day</div>
                <div class="metric-value">{:.2f} kg CO₂</div>
            </div>
            """.format(predictions_df['Predicted_Carbon_Footprint'].sum()), unsafe_allow_html=True)

        # Precise Carbon Footprint Visualization
        st.markdown("#### Daily Carbon Footprint (Precise Values)")
        fig2, ax2 = plt.subplots(figsize=(12, 4))

        # Plot with data points and values
        sns.lineplot(data=predictions_df, x='Date', y='Predicted_Carbon_Footprint',
                     color='#2e8b57', linewidth=2.5, ax=ax2)
        plt.scatter(predictions_df['Date'], predictions_df['Predicted_Carbon_Footprint'],
                    color='#2e8b57', s=100, zorder=5)

        # Add precise value labels
        for i, row in predictions_df.iterrows():
            ax2.text(row['Date'], row['Predicted_Carbon_Footprint'],
                     f'{row["Predicted_Carbon_Footprint"]:.2f}',
                     ha='center', va='bottom', fontsize=10)

        # Set y-axis limits for better zoom
        cf_min = predictions_df['Predicted_Carbon_Footprint'].min()
        cf_max = predictions_df['Predicted_Carbon_Footprint'].max()
        ax2.set_ylim(cf_min - 0.5, cf_max + 0.5)

        ax2.set_ylabel('kg CO₂ equivalent', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

        # Cigarettes Information
        st.markdown("### 🚬 Health Impact Estimation")
        avg_cigarettes = predictions_df['Cigarettes_per_Day'].mean()
        st.markdown(f"""
        <div class="info-box">
            <h4>Equivalent Cigarette Exposure</h4>
            <p>Based on the predicted air quality, breathing this air is equivalent to smoking approximately:</p>
            <p style="font-size:24px; text-align:center; font-weight:bold; margin:10px 0;">
                {avg_cigarettes:.1f} cigarettes per day 🚬
            </p>
            <p><small>Note: This estimates the health impact of particulate pollution in terms of equivalent cigarette consumption.</small></p>
        </div>
        """, unsafe_allow_html=True)

        # Map Visualization
        st.markdown("### 📍 Location-based AQI Zones")
        m = folium.Map(location=[18.5204, 73.8567], zoom_start=12, tiles='cartodbpositron')
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in predictions_df.iterrows():
            folium.Marker(
                locations[location],
                popup=f"""
                    <b>Date:</b> {row['Date'].date()}<br>
                    <b>AQI:</b> {row['Predicted_AQI']:.1f}<br>
                    <b>Zone:</b> {row['AQI_Zone']}<br>
                    <b>CO₂:</b> {row['Predicted_Carbon_Footprint']:.2f} kg<br>
                    <b>Cig Equiv:</b> {row['Cigarettes_per_Day']:.1f}/day
                """,
                icon=folium.Icon(color='red' if row['AQI_Zone'] == 'Red' else
                'orange' if row['AQI_Zone'] == 'Yellow' else 'green')
            ).add_to(marker_cluster)

        folium_static(m, width=800)

    elif 'predictions_df' in st.session_state and st.session_state.predictions_df is None:
        st.error("❌ No predictions available for this location.")

with tab2:
    st.header("📧 Email Alert System")
    st.write("Send AQI alerts and forecasts via email to stakeholders")

    with st.form("email_form"):
        st.subheader("Email Configuration")

        col1, col2 = st.columns(2)
        with col1:
            receiver_email = st.text_input("Recipient Email", "recipient@example.com", key="recipient_email")
            email_subject = st.text_input("Subject", "AQI Alert Notification", key="email_subject")

        with col2:
            smtp_server = st.text_input("SMTP Server", "smtp.gmail.com", key="smtp_server")
            smtp_port = st.number_input("SMTP Port", 587, key="smtp_port")

        sender_email = st.text_input("Your Email", "your.email@gmail.com", key="sender_email")
        sender_password = st.text_input("Your Password", type="password", key="sender_password")

        # Create message template
        default_message = f"""AQI Alert Notification:

Location: {location if 'predictions_df' in st.session_state else 'Not specified'}

"""
        if 'predictions_df' in st.session_state and st.session_state.predictions_df is not None:
            default_message += "Predicted AQI values for the next 7 days:\n"
            for _, row in st.session_state.predictions_df.iterrows():
                default_message += f"{row['Date'].date()}: AQI {row['Predicted_AQI']:.0f} ({row['AQI_Zone']} Zone)\n"
            default_message += f"\nAverage Carbon Footprint: {st.session_state.predictions_df['Predicted_Carbon_Footprint'].mean():.2f} kg CO₂ per day"
        else:
            default_message += "No current prediction data available. Please generate a forecast first."

        email_message = st.text_area("Message", default_message, height=200, key="email_message")

        if st.form_submit_button("Send Email Alert"):
            if all([receiver_email, smtp_server, sender_email, sender_password]):
                if send_alert_email(receiver_email, email_subject, email_message,
                                    smtp_server, smtp_port, sender_email, sender_password):
                    st.success("✅ Email sent successfully!")
                else:
                    st.error("❌ Failed to send email. Please check your settings.")
            else:
                st.error("Please fill all required fields")

    st.markdown("""
    <div class="info-box">
        <h4>Email Configuration Tips</h4>
        <ul>
            <li>For Gmail, you may need to enable "Less secure app access" or create an "App Password"</li>
            <li>Common SMTP servers:
                <ul>
                    <li>Gmail: smtp.gmail.com (port 587)</li>
                    <li>Outlook: smtp-mail.outlook.com (port 587)</li>
                    <li>Yahoo: smtp.mail.yahoo.com (port 465 or 587)</li>
                </ul>
            </li>
            <li>The message will automatically include prediction data if available</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


