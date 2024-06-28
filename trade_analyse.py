import json
import numpy as np
import pandas as pd

def load_trades(file_path):
    try:
        with open(file_path) as f:
            trades = json.load(f)
        return pd.DataFrame(trades["data"])
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None

# ROI = (Total PnL / Initial Capital) * 100
def calc_roi(close_trades, initial_capital):
    total_pnl = close_trades['totalPnl'].sum()
    roi = (total_pnl / initial_capital) * 100
    return roi

# Win Rate = (Number of Winning Trades / Total Number of Trades) * 100
def calc_win_rate(close_trades):
    wins = close_trades[close_trades['totalPnl'] > 0].shape[0]
    total_trades = close_trades.shape[0]
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    return win_rate

# Sharpe Ratio = (Average Daily Return - Risk-Free Rate) / Standard Deviation of Daily Returns
def calc_sharpe_ratio(close_trades, initial_capital, risk_free_rate_annual = 0.05):
    close_trades.loc[:, 'return'] = close_trades['totalPnl'] / initial_capital
    average_return = close_trades['return'].mean()
    risk_free_rate_daily = risk_free_rate_annual / 252
    std_dev_returns = close_trades['return'].std()
    sharpe_ratio = (average_return - risk_free_rate_daily) / std_dev_returns
    return sharpe_ratio

# Maximum Drawdown (MDD) = (Portfolio Value / Running Maximum) - 1
def calc_drawdown(close_trades, initial_capital):
    close_trades.loc[:, 'cumulativePnl'] = close_trades['totalPnl'].cumsum()
    close_trades.loc[:, 'portfolioValue'] = initial_capital + close_trades['cumulativePnl']
    close_trades.loc[:, 'runningMax'] = close_trades['portfolioValue'].cummax()
    close_trades.loc[:, 'drawdown'] = close_trades['portfolioValue'] / close_trades['runningMax'] - 1
    mdd = close_trades['drawdown'].min()
    return mdd

# Odds Ratio = Number of Winning Trades / Number of Losing Trades
def calc_odds_ratio(close_trades):
    wins = close_trades[close_trades['totalPnl'] > 0].shape[0]
    losses = close_trades[close_trades['totalPnl'] <= 0].shape[0]
    odds_ratio = wins / losses if losses != 0 else float('inf')
    return odds_ratio

# Profit Factor = Gross Profit / Gross Loss
def calc_profit_factor(close_trades):
    gross_profit = close_trades[close_trades['totalPnl'] > 0]['totalPnl'].sum()
    gross_loss = -close_trades[close_trades['totalPnl'] <= 0]['totalPnl'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    return profit_factor

if __name__ == "__main__":
    initial_capital = 8000.0
    df = load_trades('trades.json')
    
    if df is not None:
        # Filter only close transactions (Close Long, Close Short)
        close_trades = df[df['side'].str.contains('sell', case=False)].copy()  # Explicitly create a copy

        # calc total PnL including fees
        close_trades.loc[:, 'totalPnl'] = close_trades['fillPnl'].astype(float) + close_trades['fee'].astype(float)

        roi = calc_roi(close_trades, initial_capital)
        print(f"ROI: {roi:.2f}%")

        win_rate = calc_win_rate(close_trades)
        print(f"Win Rate: {win_rate:.2f}%")

        mdd = calc_drawdown(close_trades, initial_capital)
        print(f"Maximum Drawdown (MDD): {mdd:.2%}")

        odds_ratio = calc_odds_ratio(close_trades)
        print(f"Odds Ratio: {odds_ratio:.2f}")

        profit_factor = calc_profit_factor(close_trades)
        print(f"Profit Factor: {profit_factor:.2f}")

        sharpe_ratio = calc_sharpe_ratio(close_trades, initial_capital)
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        print("Failed to load trades. Exiting program.")
