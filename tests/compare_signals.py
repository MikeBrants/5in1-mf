"""
Compare Pine Script signals with Python implementation.
Uses exported CSV from TradingView with Pine indicator values.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '..')

from indicators.mama_fama import compute_mama_fama
from indicators.kama import compute_kama
from indicators.ichimoku import compute_ichimoku_signals


def load_pine_data(filepath: str) -> pd.DataFrame:
    """Load and prepare Pine Script exported data."""
    df = pd.read_csv(filepath)
    df.columns = [col.strip() for col in df.columns]
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('datetime', inplace=True)
    return df


def compare_mama_fama(df: pd.DataFrame, tolerance: float = 0.01) -> dict:
    """Compare MAMA/FAMA/KAMA values between Pine and Python."""
    mama_py, fama_py = compute_mama_fama(df['close'].values, fast_limit=0.5, slow_limit=0.05)
    kama_py = compute_kama(df['close'].values, period=10, fast_period=2, slow_period=30)
    
    mama_pine = df['MAMA'].values
    fama_pine = df['FAMA'].values
    kama_pine = df['KAMA'].values
    
    warmup = 50
    
    mama_diff = np.abs(mama_py[warmup:] - mama_pine[warmup:]) / mama_pine[warmup:] * 100
    fama_diff = np.abs(fama_py[warmup:] - fama_pine[warmup:]) / fama_pine[warmup:] * 100
    kama_diff = np.abs(kama_py[warmup:] - kama_pine[warmup:]) / kama_pine[warmup:] * 100
    
    return {
        'mama': {'mean_diff_pct': np.nanmean(mama_diff), 'max_diff_pct': np.nanmax(mama_diff), 'within_tolerance': np.nanmean(mama_diff < tolerance * 100) * 100},
        'fama': {'mean_diff_pct': np.nanmean(fama_diff), 'max_diff_pct': np.nanmax(fama_diff), 'within_tolerance': np.nanmean(fama_diff < tolerance * 100) * 100},
        'kama': {'mean_diff_pct': np.nanmean(kama_diff), 'max_diff_pct': np.nanmax(kama_diff), 'within_tolerance': np.nanmean(kama_diff < tolerance * 100) * 100}
    }


def compare_signals(df: pd.DataFrame) -> dict:
    """Compare buy/sell signals between Pine and Python."""
    mama_py, fama_py = compute_mama_fama(df['close'].values)
    
    buy_py = np.zeros(len(df), dtype=int)
    sell_py = np.zeros(len(df), dtype=int)
    
    for i in range(1, len(df)):
        if mama_py[i] > fama_py[i] and mama_py[i-1] <= fama_py[i-1]:
            buy_py[i] = 1
        if mama_py[i] < fama_py[i] and mama_py[i-1] >= fama_py[i-1]:
            sell_py[i] = 1
    
    buy_pine = df['Buy Signal'].values
    sell_pine = df['Sell Signal'].values
    
    return {
        'buy_signals': {
            'python_total': int(np.sum(buy_py)),
            'pine_total': int(np.sum(buy_pine)),
            'matched': int(np.sum((buy_py == 1) & (buy_pine == 1))),
            'match_rate': np.sum((buy_py == 1) & (buy_pine == 1)) / max(np.sum(buy_pine), 1) * 100
        },
        'sell_signals': {
            'python_total': int(np.sum(sell_py)),
            'pine_total': int(np.sum(sell_pine)),
            'matched': int(np.sum((sell_py == 1) & (sell_pine == 1))),
            'match_rate': np.sum((sell_py == 1) & (sell_pine == 1)) / max(np.sum(sell_pine), 1) * 100
        }
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare Pine vs Python signals')
    parser.add_argument('--file', type=str, default='data/BYBIT_BTCUSDT-60.csv')
    args = parser.parse_args()
    
    print(f"Loading data from {args.file}...")
    df = load_pine_data(args.file)
    print(f"Loaded {len(df)} candles")
    
    print("\nComparing indicators...")
    indicator_results = compare_mama_fama(df)
    signal_results = compare_signals(df)
    
    print("\n" + "=" * 50)
    print("PINE vs PYTHON COMPARISON")
    print("=" * 50)
    
    for name, data in indicator_results.items():
        print(f"\n{name.upper()}: Mean diff {data['mean_diff_pct']:.4f}% | Within 1%: {data['within_tolerance']:.1f}%")
    
    for sig, data in signal_results.items():
        print(f"\n{sig.upper()}: Python {data['python_total']} | Pine {data['pine_total']} | Match {data['match_rate']:.1f}%")


if __name__ == '__main__':
    main()
