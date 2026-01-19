"""
Ichimoku Cloud indicator and signal generation.
"""
import numpy as np
from numba import njit


@njit
def compute_ichimoku_signals(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26
) -> tuple:
    """
    Compute Ichimoku components and generate buy/sell signals.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        tenkan_period: Tenkan-sen (Conversion Line) period
        kijun_period: Kijun-sen (Base Line) period
        senkou_b_period: Senkou Span B period
        displacement: Cloud displacement (chikou/senkou shift)
    
    Returns:
        Tuple of (buy_signals, sell_signals) arrays
    """
    n = len(close)
    
    # Initialize arrays
    tenkan = np.zeros(n)
    kijun = np.zeros(n)
    senkou_a = np.zeros(n)
    senkou_b = np.zeros(n)
    chikou = np.zeros(n)
    
    buy_signals = np.zeros(n, dtype=np.int64)
    sell_signals = np.zeros(n, dtype=np.int64)
    
    # Helper function for period high/low
    def period_high(arr, end, period):
        start = max(0, end - period + 1)
        max_val = arr[start]
        for i in range(start, end + 1):
            if arr[i] > max_val:
                max_val = arr[i]
        return max_val
    
    def period_low(arr, end, period):
        start = max(0, end - period + 1)
        min_val = arr[start]
        for i in range(start, end + 1):
            if arr[i] < min_val:
                min_val = arr[i]
        return min_val
    
    # Compute Ichimoku components
    for i in range(n):
        # Tenkan-sen (Conversion Line)
        if i >= tenkan_period - 1:
            tenkan[i] = (period_high(high, i, tenkan_period) + period_low(low, i, tenkan_period)) / 2
        else:
            tenkan[i] = close[i]
        
        # Kijun-sen (Base Line)
        if i >= kijun_period - 1:
            kijun[i] = (period_high(high, i, kijun_period) + period_low(low, i, kijun_period)) / 2
        else:
            kijun[i] = close[i]
        
        # Senkou Span A (Leading Span A) - displaced forward
        senkou_a[i] = (tenkan[i] + kijun[i]) / 2
        
        # Senkou Span B (Leading Span B) - displaced forward
        if i >= senkou_b_period - 1:
            senkou_b[i] = (period_high(high, i, senkou_b_period) + period_low(low, i, senkou_b_period)) / 2
        else:
            senkou_b[i] = close[i]
        
        # Chikou Span (Lagging Span) - displaced backward
        if i >= displacement:
            chikou[i] = close[i]
    
    # Generate signals based on Ichimoku conditions
    for i in range(displacement + kijun_period, n):
        # Cloud values at current position (displaced forward by 26)
        cloud_idx = i - displacement
        if cloud_idx < 0:
            continue
        
        cloud_top = max(senkou_a[cloud_idx], senkou_b[cloud_idx])
        cloud_bottom = min(senkou_a[cloud_idx], senkou_b[cloud_idx])
        
        # Previous bar conditions
        prev_cloud_idx = cloud_idx - 1
        if prev_cloud_idx < 0:
            continue
        prev_cloud_top = max(senkou_a[prev_cloud_idx], senkou_b[prev_cloud_idx])
        
        # BUY Signal conditions:
        # 1. Tenkan-sen crosses above Kijun-sen
        # 2. Price is above the cloud
        # 3. Chikou Span is above price from 26 periods ago
        tenkan_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        price_above_cloud = close[i] > cloud_top
        chikou_above = close[i] > close[i - displacement] if i >= displacement else False
        
        if tenkan_cross_up and price_above_cloud and chikou_above:
            buy_signals[i] = 1
        
        # SELL Signal conditions:
        # 1. Tenkan-sen crosses below Kijun-sen
        # 2. Price is below the cloud
        # 3. Chikou Span is below price from 26 periods ago
        tenkan_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        price_below_cloud = close[i] < cloud_bottom
        chikou_below = close[i] < close[i - displacement] if i >= displacement else False
        
        if tenkan_cross_down and price_below_cloud and chikou_below:
            sell_signals[i] = 1
    
    return buy_signals, sell_signals
