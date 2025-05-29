import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import mplfinance as mpf

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file path
file_path = os.path.join(script_dir, 'asset_portfolio.csv')

# Load data
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    exit("File not found. Please check the file path.")
except Exception as e:
    exit(f"An error occurred: {e}")

# Convert date to datetime and sort
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y', dayfirst=True)
data = data.sort_values('date')

# Add RSI
def calculate_rsi(data, period=14):
    delta = data['close'].diff()  # Calculate daily price changes
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # Average gain
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # Average loss
    rs = gain / loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    return rsi

data['RSI_14'] = calculate_rsi(data, period=14)

# Add SMA
def calculate_sma(data, period=50):
    sma = data['close'].rolling(window=period).mean()  # Rolling mean
    return sma

data['SMA_50'] = calculate_sma(data, period=50)

# Add Volatility
def calculate_volatility(data, period=30):
    daily_returns = data['close'].pct_change()  # Daily returns
    volatility = daily_returns.rolling(window=period).std()  # Standard deviation
    return volatility

data['volatility_30'] = calculate_volatility(data, period=30)

# Add MACD (Moving Average Convergence Divergence)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()  # Short-term EMA
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()  # Long-term EMA
    macd = short_ema - long_ema  # MACD line
    signal = macd.ewm(span=signal_window, adjust=False).mean()  # Signal line
    histogram = macd - signal  # MACD histogram
    return macd, signal, histogram

data['MACD'], data['MACD_signal'], data['MACD_histogram'] = calculate_macd(data)

# Add Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data['close'].rolling(window=window).mean()  # Simple Moving Average
    rolling_std = data['close'].rolling(window=window).std()  # Rolling Standard Deviation
    upper_band = sma + (rolling_std * num_std)  # Upper Bollinger Band
    lower_band = sma - (rolling_std * num_std)  # Lower Bollinger Band
    bandwidth = upper_band - lower_band  # Bollinger Bandwidth
    return upper_band, lower_band, bandwidth

data['Bollinger_Upper'], data['Bollinger_Lower'], data['Bollinger_Bandwidth'] = calculate_bollinger_bands(data)

# Add ATR (Average True Range)
def calculate_atr(data, period=14):
    high_low = data['high'] - data['low']  # High - Low
    high_close = abs(data['high'] - data['close'].shift(1))  # |High - Previous Close|
    low_close = abs(data['low'] - data['close'].shift(1))  # |Low - Previous Close|
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)  # Max of the three
    atr = true_range.rolling(window=period).mean()  # Average True Range
    return atr

data['ATR_14'] = calculate_atr(data, period=14)

# Add Momentum
def calculate_momentum(data, period=5):
    momentum = data['close'].diff(period)  # Momentum over a period
    return momentum

data['Momentum_5'] = calculate_momentum(data, period=5)

# Add Volume Moving Average
def calculate_volume_ma(data, period=5):
    volume_ma = data['volume'].rolling(window=period).mean()  # Volume Moving Average
    return volume_ma

data['Volume_MA_5'] = calculate_volume_ma(data, period=5)

# Handle missing data
data = data.dropna()

# Save to new file
output_path = 'asset_portfolio_with_featuresv2.csv'
data.to_csv(output_path, index=False)
print(f"File saved to {output_path}")


# --- Visualization Section ---
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
from scipy import interpolate
import numpy as np

def generate_visualizations(data, last_n_days=100):
    """Generate all plots with Bollinger Bands in green/red"""
    vis_data = data.set_index('date').tail(last_n_days).copy()
    
    # --- Smoothing Helper Function ---
    def smooth_line(x, y, num_points=1000):
        """Handle duplicates and smooth data"""
        df = pd.DataFrame({'x': mdates.date2num(x), 'y': y})
        df = df.groupby('x').mean().reset_index()  # Aggregate duplicates
        if len(df) >= 4:
            spline = interpolate.make_interp_spline(df['x'], df['y'], k=3)
            x_smooth = np.linspace(df['x'].min(), df['x'].max(), num_points)
            y_smooth = spline(x_smooth)
            return mdates.num2date(x_smooth), y_smooth
        return x, y

    # Create output directory
    os.makedirs('visualizations', exist_ok=True)

    # 1. RSI Plot (unchanged)
    plt.figure(figsize=(12, 4))
    x_smooth, y_smooth = smooth_line(vis_data.index, vis_data['RSI_14'])
    plt.plot(x_smooth, y_smooth, label='RSI 14', color='purple', linewidth=2)
    plt.axhline(70, linestyle='--', color='red', alpha=0.5, label='Overbought (70)')
    plt.axhline(30, linestyle='--', color='green', alpha=0.5, label='Oversold (30)')
    plt.fill_between(x_smooth, y_smooth, 70, where=(y_smooth>=70), color='red', alpha=0.1)
    plt.fill_between(x_smooth, y_smooth, 30, where=(y_smooth<=30), color='green', alpha=0.1)
    plt.title('RSI 14-Day Oscillator', pad=20)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/RSI_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. BOLLINGER BANDS (UPDATED COLORS)
    ap = [
        mpf.make_addplot(vis_data['Bollinger_Upper'], color='green', alpha=0.7, width=1.5),
        mpf.make_addplot(vis_data['Bollinger_Lower'], color='red', alpha=0.7, width=1.5),
        mpf.make_addplot(vis_data['close'], color='black', width=1.2)
    ]
    mpf.plot(
        vis_data,
        type='candle',
        style='charles',
        addplot=ap,
        title='Bollinger Bands (20-day, 2σ)',
        savefig='visualizations/bollinger_bands.png',
        figscale=1.2,
        volume=False,
        fill_between=dict(
            y1=vis_data['Bollinger_Upper'].values,
            y2=vis_data['Bollinger_Lower'].values,
            color='lightgray',
            alpha=0.2
        )
    )

    # 3. MACD Plot (unchanged)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), 
                                  gridspec_kw={'height_ratios': [2, 1]},
                                  sharex=True)
    x_smooth, y_smooth = smooth_line(vis_data.index, vis_data['close'])
    ax1.plot(x_smooth, y_smooth, label='Price', color='black', linewidth=1.5)
    ax1.set_title('Price with MACD Indicator', pad=20)
    ax1.grid(alpha=0.3)
    x_macd, macd_smooth = smooth_line(vis_data.index, vis_data['MACD'])
    _, signal_smooth = smooth_line(vis_data.index, vis_data['MACD_signal'])
    ax2.plot(x_macd, macd_smooth, label='MACD', color='blue', linewidth=1.2)
    ax2.plot(x_macd, signal_smooth, label='Signal', color='orange', linewidth=1.2)
    hist_colors = np.where(vis_data['MACD_histogram'] >= 0, 'green', 'red')
    ax2.bar(vis_data.index, vis_data['MACD_histogram'], 
            color=hist_colors, alpha=0.3, width=0.8)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(alpha=0.3)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('visualizations/MACD_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Volatility Plot (unchanged)
    plt.figure(figsize=(12, 4))
    x_smooth, atr_smooth = smooth_line(vis_data.index, vis_data['ATR_14'])
    _, vol_smooth = smooth_line(vis_data.index, vis_data['volatility_30'])
    plt.fill_between(x_smooth, atr_smooth, alpha=0.2, color='red', label='14-day ATR')
    plt.plot(x_smooth, atr_smooth, color='red', linewidth=1.2)
    plt.plot(x_smooth, vol_smooth, label='30-day Volatility', color='green', linewidth=1.5)
    plt.title('Volatility Metrics Comparison', pad=20)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/volatility_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 5. Momentum and Volume (unchanged)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    x_smooth, mom_smooth = smooth_line(vis_data.index, vis_data['Momentum_5'])
    ax1.plot(x_smooth, mom_smooth, label='5-day Momentum', color='navy', linewidth=1.5)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.fill_between(x_smooth, mom_smooth, 0, 
                    where=(mom_smooth>=0), color='green', alpha=0.1)
    ax1.fill_between(x_smooth, mom_smooth, 0, 
                    where=(mom_smooth<0), color='red', alpha=0.1)
    ax1.set_title('Price Momentum & Volume Trends', pad=20)
    ax1.grid(alpha=0.3)
    ax2.bar(vis_data.index, vis_data['volume'], 
            color=np.where(vis_data['Momentum_5']>=0, 'green', 'red'), 
            alpha=0.3, width=0.8)
    x_vol, vol_ma_smooth = smooth_line(vis_data.index, vis_data['Volume_MA_5'])
    ax2.plot(x_vol, vol_ma_smooth, label='5-day Volume MA', color='red', linewidth=1.5)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('visualizations/momentum_volume_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate visualizations
generate_visualizations(data)
