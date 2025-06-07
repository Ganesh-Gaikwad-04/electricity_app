from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import io
import base64

plt.style.use('seaborn-v0_8-darkgrid')

app = Flask(__name__)

def load_data():
    df = pd.read_csv(
        "enhanced_electricity_consumption.csv",
        usecols=["DateTime", "Consumption_kWh"]
    )
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.set_index('DateTime')
    # Resample to daily sum (total daily consumption)
    df = df.resample('D').sum()
    df = df.rename(columns={"Consumption_kWh": "Global_active_power"})
    df = df.interpolate()
    return df

data = load_data()

# Prophet Forecast
def prophet_forecast(df, days):
    df_prophet = df.reset_index().rename(columns={"DateTime": "ds", "Global_active_power": "y"})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)

# Improved ARIMA Forecast
def arima_forecast(df, days):
    y = df['Global_active_power']
    model = ARIMA(y, order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    return forecast

# Improved SARIMA Forecast
def sarima_forecast(df, days):
    y = df['Global_active_power']
    model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,7))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=days)
    return forecast

# Improved Plot forecast result
def plot_forecast(original, forecast, label):
    plt.figure(figsize=(10, 5))
    plt.plot(original[-60:], label='Actual', marker='o', color='tab:blue')
    forecast_index = pd.date_range(original.index[-1], periods=len(forecast)+1, freq='D')[1:]
    plt.plot(forecast_index, forecast, label=label + ' Forecast', marker='o', color='tab:orange')
    plt.title('Forecasted vs Actual Energy Consumption')
    plt.xlabel('Date')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# Plot historical consumption
def plot_historical(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df[-180:], color='green', label='Last 180 Days')
    plt.title('Historical Daily Energy Consumption')
    plt.xlabel('Date')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.legend()
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# Plot year-wise averages and 2026 forecast
def plot_yearly_and_forecast(df, forecast_2026):
    plt.figure(figsize=(12, 6))
    years = list(range(2011, 2026))
    yearly_means = []
    for year in years:
        year_data = df[df.index.year == year]
        if not year_data.empty:
            yearly_means.append(year_data['Global_active_power'].mean())
        else:
            yearly_means.append(np.nan)
    plt.bar([str(y) for y in years], yearly_means, color='skyblue', label='Yearly Avg (2011-2025)')
    # Forecast for 2026
    plt.bar('2026', np.mean(forecast_2026), color='orange', label='Forecast 2026')
    plt.xlabel('Year')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.title('Year-wise Average Energy Consumption and 2026 Forecast')
    plt.legend()
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# Generate insights
def generate_insights(df, forecast_values):
    current_avg = df[-30:].mean()[0]
    future_avg = np.mean(forecast_values)
    change = ((future_avg - current_avg) / current_avg) * 100
    trend = "increase" if change > 0 else "decrease"
    return f"Forecast shows a {abs(change):.2f}% {trend} in energy consumption over the next forecasted period."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    days = int(request.form['days'])
    model_type = request.form['model']

    historical_plot = plot_historical(data)
    insights = ""
    forecast_days_2026 = 366  # Leap year

    if model_type == 'prophet':
        forecast_df = prophet_forecast(data, forecast_days_2026)
        forecast_values = forecast_df['yhat'].values
        forecast_dates = forecast_df['ds'].dt.strftime('%Y-%m-%d').tolist()
        plot_url = plot_forecast(data, forecast_values, 'Prophet')
        insights = generate_insights(data, forecast_values)
    elif model_type == 'arima':
        forecast_values = arima_forecast(data, forecast_days_2026)
        forecast_dates = pd.date_range(data.index[-1], periods=forecast_days_2026+1, freq='D')[1:].strftime('%Y-%m-%d').tolist()
        plot_url = plot_forecast(data, forecast_values, 'ARIMA')
        insights = generate_insights(data, forecast_values)
    elif model_type == 'sarima':
        forecast_values = sarima_forecast(data, forecast_days_2026)
        forecast_dates = pd.date_range(data.index[-1], periods=forecast_days_2026+1, freq='D')[1:].strftime('%Y-%m-%d').tolist()
        plot_url = plot_forecast(data, forecast_values, 'SARIMA')
        insights = generate_insights(data, forecast_values)
    else:
        return jsonify({'error': 'Invalid model type'})

    yearly_plot = plot_yearly_and_forecast(data, forecast_values)

    return render_template(
        'result.html',
        plot_url=plot_url,
        historical_plot=historical_plot,
        yearly_plot=yearly_plot,
        insights=insights,
        forecast=zip(forecast_dates, forecast_values)
    )

if __name__ == '__main__':
    app.run(debug=True)