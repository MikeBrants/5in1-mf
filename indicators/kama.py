"""
Kaufman Adaptive Moving Average (KAMA)
Based on Perry Kaufman's algorithm.
"""
import numpy as np
from numba import njit


@njit
def compute_kama(close: np.ndarray, period: int = 10, fast_period: int = 2, slow_period: int = 30) -> np.ndarray:
    """
    Compute Kaufman Adaptive Moving Average.
    
    Args:
        close: Array of closing prices
        period: Efficiency ratio period (default 10)
        fast_period: Fast smoothing constant period (default 2)
        slow_period: Slow smoothing constant period (default 30)
    
    Returns:
        KAMA array
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA with first close
    kama[0] = close[0]
    
    for i in range(1, n):
        if i < period:
            kama[i] = close[i]
            continue
        
        # Change: absolute price change over period
        change = abs(close[i] - close[i - period])
        
        # Volatility: sum of absolute daily changes
        volatility = 0.0
        for j in range(period):
            volatility += abs(close[i - j] - close[i - j - 1])
        
        # Efficiency Ratio
        if volatility != 0:
            er = change / volatility
        else:
            er = 0
        
        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama
