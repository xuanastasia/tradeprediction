# Automated Portfolio Optimization

## Project Description
This project designs an algorithmic trading strategy that combines Bitcoin with traditional assets to build risk-adjusted, high-return portfolios. By leveraging Bitcoin’s low market correlation and applying machine learning techniques, we optimize asset allocation to enhance returns while managing volatility. The goal is to provide a data-driven framework for resilient portfolio construction, supporting smarter investment decisions in evolving markets.

## Portfolio Composition
| Category        | Assets       | Purpose                     |
|-----------------|--------------|-----------------------------|
| Growth         | BTC-USD, QQQ | Capital appreciation       |
| Hedging        | GLD, BND     | Volatility dampening       |
| Stable Income  | SCHV         | Consistent yield           |

## Allocation Logic
## Kelly Criterion implementation
allocation = {
    'LSTM': 0.6,    # 60% allocation
    'RF': 0.3,      # 30% allocation  
    'B&H': 0.1      # 10% allocation
}

## Trading Strategy
![deepseek_mermaid_20250529_329f95](https://github.com/user-attachments/assets/b8b56954-cb7c-4aa1-a839-aee2c6d15ffb)

## Performance Metrics

### Backtest Results (Jan 1 - Feb 28, 2025)

| Metric                  | LSTM Model   | Random Forest | Buy & Hold | SPY Benchmark |
|-------------------------|--------------|---------------|------------|---------------|
| **Sharpe Ratio**        | 4.34         | 2.76          | 0.57       | 1.02          |
| **Annual Return**       | 33.6%        | 28.1%         | 5.6%       | 12.4%         |
| **Max Drawdown**        | -1.30%       | -2.89%        | -4.58%     | -6.12%        |
| **Annual Volatility**   | 7.74%        | 10.17%        | 9.88%      | 15.23%        |
| **Win Rate**            | 78%          | 65%           | N/A        | N/A           |
| **Avg Trade Duration**  | 6.2 days     | 2.1 days      | N/A        | N/A           |

### Stress Test Performance (Mar 1 - Apr 4, 2025)

```diff
+ Portfolio (60% LSTM / 30% RF / 10% B&H): -6.37% 
- SPY Benchmark: -14.90% 
! BTC-USD: -22.34%
```

# Installation
## Clone repository
git clone https://github.com/yourusername/portfolio-optimization.git

## Install dependencies
cd portfolio-optimization
pip install -r requirements.txt

## Usage
# Run full backtest
python main.py --start 20250101 --end 20250228

# Train specific model
python train.py --model lstm --ticker BTC-USD


