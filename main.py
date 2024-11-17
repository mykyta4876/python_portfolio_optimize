import pandas as pd
import numpy as np
from scipy.optimize import minimize
import xgboost as xgb

# Load trade data (replace this with actual data loading logic)
data = pd.read_csv('CS_-_Tom_Peetoom_List_of_Trades_2024-09-16.csv')

# Calculate daily returns from the trade data
def calculate_returns(data):
    data['Daily_Return'] = data['Cum. Profit USD'].pct_change().fillna(0)
    return data['Daily_Return']

# Function to calculate portfolio drawdown
def calculate_drawdown(returns):
    cum_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

# Function to optimize portfolio for max return and restricted drawdown
def optimize_portfolio(returns, max_drawdown):
    n = len(returns.columns)
    
    def portfolio_return(weights):
        return np.dot(weights, returns.mean())

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))

    def objective(weights):
        return -portfolio_return(weights) / portfolio_volatility(weights)
    
    def constraint_drawdown(weights):
        portfolio_returns = np.dot(returns, weights)
        drawdown = calculate_drawdown(portfolio_returns)
        return max_drawdown - abs(drawdown)

    # Initial guess: equal weights
    initial_weights = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = ({'type': 'ineq', 'fun': constraint_drawdown},
                   {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Preparing the data
data['Daily_Return'] = calculate_returns(data)
X = data[['Price USD', 'Contracts', 'Cum. Profit USD', 'Run-up USD', 'Drawdown USD']]
y = data['Daily_Return']

# Train XGBoost model to predict future returns
model = xgb.XGBRegressor()
model.fit(X, y)

# Predict returns
predicted_returns = model.predict(X)

# Optimize portfolio
max_allowed_drawdown = 0.10
weights = optimize_portfolio(data[['Daily_Return']], max_allowed_drawdown)

# Print optimized weights
print("Optimized portfolio weights:", weights)

# Simulate daily returns for the optimized portfolio
# portfolio_returns = np.dot(data['Daily_Return'], weights)
"""
It seems like the portfolio contains only one asset, which is why the weights array is [1.]. In this case, you don't need to use np.dot, since you're essentially multiplying each return by 1.

You can resolve this by directly assigning the daily returns as the portfolio returns, since there's only one asset:
"""
portfolio_returns = data['Daily_Return']

cumulative_returns = np.cumsum(portfolio_returns)
if not cumulative_returns.empty:
    print("Cumulative portfolio returns:", cumulative_returns.iloc[-1])
else:
    print("Cumulative returns are empty.")
