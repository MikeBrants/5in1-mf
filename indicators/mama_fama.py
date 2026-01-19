"""
MESA Adaptive Moving Average (MAMA) and Following Adaptive Moving Average (FAMA)
Based on John Ehlers' algorithm.
"""
import numpy as np
from numba import njit


@njit
def compute_mama_fama(close: np.ndarray, fast_limit: float = 0.5, slow_limit: float = 0.05) -> tuple:
    """
    Compute MAMA and FAMA using Ehlers' Hilbert Transform.
    
    Args:
        close: Array of closing prices
        fast_limit: Fast alpha limit (default 0.5)
        slow_limit: Slow alpha limit (default 0.05)
    
    Returns:
        Tuple of (mama, fama) arrays
    """
    n = len(close)
    
    # Initialize arrays
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    period = np.zeros(n)
    q1 = np.zeros(n)
    i1 = np.zeros(n)
    ji = np.zeros(n)
    jq = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re = np.zeros(n)
    im = np.zeros(n)
    phase = np.zeros(n)
    mama = np.zeros(n)
    fama = np.zeros(n)
    
    # Initialize period
    for i in range(n):
        period[i] = 6.0
    
    for i in range(6, n):
        # Smooth price
        smooth[i] = (4 * close[i] + 3 * close[i-1] + 2 * close[i-2] + close[i-3]) / 10.0
        
        # Detrender
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] - 
                       0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)
        
        # Compute InPhase and Quadrature components
        q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] - 
                0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
        i1[i] = detrender[i-3]
        
        # Advance phase by 90 degrees
        ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] - 
                0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * period[i-1] + 0.54)
        jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] - 
                0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * period[i-1] + 0.54)
        
        # Phasor addition for 3-bar averaging
        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]
        
        # Smooth I and Q components
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]
        
        # Homodyne discriminator
        re[i] = i2[i] * i2[i-1] + q2[i] * q2[i-1]
        im[i] = i2[i] * q2[i-1] - q2[i] * i2[i-1]
        
        re[i] = 0.2 * re[i] + 0.8 * re[i-1]
        im[i] = 0.2 * im[i] + 0.8 * im[i-1]
        
        # Compute period
        if im[i] != 0 and re[i] != 0:
            period[i] = 2 * np.pi / np.arctan(im[i] / re[i])
        
        # Clamp period
        if period[i] > 1.5 * period[i-1]:
            period[i] = 1.5 * period[i-1]
        if period[i] < 0.67 * period[i-1]:
            period[i] = 0.67 * period[i-1]
        if period[i] < 6:
            period[i] = 6
        if period[i] > 50:
            period[i] = 50
        
        period[i] = 0.2 * period[i] + 0.8 * period[i-1]
        
        # Compute phase
        if i1[i] != 0:
            phase[i] = np.arctan(q1[i] / i1[i]) * 180 / np.pi
        
        # Compute delta phase
        delta_phase = phase[i-1] - phase[i]
        if delta_phase < 1:
            delta_phase = 1
        
        # Compute alpha
        alpha = fast_limit / delta_phase
        if alpha < slow_limit:
            alpha = slow_limit
        if alpha > fast_limit:
            alpha = fast_limit
        
        # Compute MAMA and FAMA
        mama[i] = alpha * close[i] + (1 - alpha) * mama[i-1]
        fama[i] = 0.5 * alpha * mama[i] + (1 - 0.5 * alpha) * fama[i-1]
    
    return mama, fama
