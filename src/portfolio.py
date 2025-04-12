import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Portfolio:

    def __init__(self,
                 initial_capital: float = 100000.0,
                 fee_rate: float = 0.0006):
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        if not (0 <= fee_rate < 1):
            raise ValueError("Fee rate must be between 0 (inclusive) and 1 (exclusive).")

        self.initial_capital: float = initial_capital
        self.fee_rate: float = fee_rate

        self.current_cash: float = initial_capital
        self.current_positions: Dict[str, float] = {}
        self.current_holdings_value: Dict[str, float] = {}

        self.history: List[Tuple[pd.Timestamp, float]] = []
        self.trade_log: List[Dict[str, Any]] = []

        logging.info(f"Portfolio initialized with Capital: ${initial_capital:,.2f}, Fee Rate: {fee_rate:.4%}")

    @property
    def current_total_value(self) -> float:
        total_holdings = sum(self.current_holdings_value.values())
        return self.current_cash + total_holdings

    def update_market_value(self, timestamp: pd.Timestamp, market_data: pd.Series) -> None:
        self.current_holdings_value.clear()
        for asset, quantity in self.current_positions.items():
            if quantity == 0:
                 continue
            try:
                current_price = market_data[asset]
                if pd.isna(current_price) or current_price <= 0:
                     logging.warning(f"[{timestamp}] Invalid or missing market price ({current_price}) for asset {asset}. Holding value might be inaccurate.")
                     market_value = 0.0
                else:
                    market_value = quantity * current_price
                self.current_holdings_value[asset] = market_value
            except KeyError:
                logging.error(f"[{timestamp}] Market price for asset '{asset}' not found in provided market_data. Cannot update its value.")
                self.current_holdings_value[asset] = 0.0

        current_total = self.current_total_value
        self.history.append((timestamp, current_total))


    def transact_asset(self,
                       timestamp: pd.Timestamp,
                       asset: str,
                       order_type: str,
                       quantity: float,
                       price: float) -> bool:
        order_type = order_type.upper()
        if quantity <= 0:
            logging.warning(f"[{timestamp}] Trade quantity must be positive. Attempted {order_type} {quantity} {asset}.")
            return False
        if price <= 0:
            logging.warning(f"[{timestamp}] Trade price must be positive. Attempted {order_type} {asset} at {price}.")
            return False
        if order_type not in ['BUY', 'SELL']:
             logging.error(f"[{timestamp}] Invalid order type '{order_type}'. Must be 'BUY' or 'SELL'.")
             return False


        trade_value = quantity * price
        fee = trade_value * self.fee_rate
        current_position = self.current_positions.get(asset, 0.0)

        if order_type == 'BUY':
            required_cash = trade_value + fee
            if self.current_cash >= required_cash:
                self.current_cash -= required_cash
                self.current_positions[asset] = current_position + quantity
                self._log_trade(timestamp, asset, 'BUY', quantity, price, fee, trade_value)
                return True
            else:
                logging.warning(f"[{timestamp}] Insufficient funds to BUY {quantity:.4f} {asset}. Required: ${required_cash:,.2f}, Available: ${self.current_cash:,.2f}")
                return False

        elif order_type == 'SELL':
            if current_position >= quantity:
                proceeds = trade_value - fee
                self.current_cash += proceeds
                self.current_positions[asset] = current_position - quantity
                if abs(self.current_positions[asset]) < 1e-9:
                    del self.current_positions[asset]

                self._log_trade(timestamp, asset, 'SELL', quantity, price, fee, trade_value)
                return True
            else:
                logging.warning(f"[{timestamp}] Insufficient position to SELL {quantity:.4f} {asset}. Holding: {current_position:.4f}, Attempted: {quantity:.4f}")
                return False

        return False

    def _log_trade(self, timestamp, asset, order_type, quantity, price, fee, trade_value):
        trade_record = {
            'timestamp': timestamp,
            'asset': asset,
            'type': order_type,
            'quantity': quantity,
            'price': price,
            'trade_value': trade_value,
            'fee': fee,
            'cash_change': -(trade_value + fee) if order_type == 'BUY' else (trade_value - fee),
            'position_change': quantity if order_type == 'BUY' else -quantity,
            'cash_after_trade': self.current_cash,
            'position_after_trade': self.current_positions.get(asset, 0.0)
        }
        self.trade_log.append(trade_record)

    def get_history_df(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame(columns=['timestamp', 'total_value']).set_index('timestamp')
        df = pd.DataFrame(self.history, columns=['timestamp', 'total_value'])
        df.set_index('timestamp', inplace=True)
        return df

    def get_trade_log_df(self) -> pd.DataFrame:
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)


if __name__ == "__main__":
    dates = pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:05:00', '2023-01-01 10:10:00', '2023-01-01 10:15:00'])
    market_prices = {
        'BTCUSD': pd.Series([20000, 20100, 20050, 20200], index=dates),
        'ETHUSD': pd.Series([1500, 1510, 1505, 1520], index=dates)
    }

    portfolio = Portfolio(initial_capital=50000, fee_rate=0.001)

    print("--- Simulation Start ---")

    ts1 = dates[0]
    prices1 = pd.Series({'BTCUSD': market_prices['BTCUSD'].loc[ts1], 'ETHUSD': market_prices['ETHUSD'].loc[ts1]})
    portfolio.update_market_value(ts1, prices1)
    print(f"\n[{ts1}] Portfolio Value: ${portfolio.current_total_value:,.2f}")
    print(f"Cash: ${portfolio.current_cash:,.2f}, Positions: {portfolio.current_positions}")

    print("\nAttempting BUY BTCUSD...")
    buy_success = portfolio.transact_asset(ts1, 'BTCUSD', 'BUY', 0.5, prices1['BTCUSD'])
    print(f"Buy executed: {buy_success}")
    print(f"After Trade - Cash: ${portfolio.current_cash:,.2f}, Positions: {portfolio.current_positions}")

    ts2 = dates[1]
    prices2 = pd.Series({'BTCUSD': market_prices['BTCUSD'].loc[ts2], 'ETHUSD': market_prices['ETHUSD'].loc[ts2]})
    portfolio.update_market_value(ts2, prices2)
    print(f"\n[{ts2}] Portfolio Value: ${portfolio.current_total_value:,.2f}")
    print(f"Cash: ${portfolio.current_cash:,.2f}, Positions: {portfolio.current_positions}")
    print(f"Holdings Value: {portfolio.current_holdings_value}")

    print("\nAttempting SELL BTCUSD...")
    sell_success = portfolio.transact_asset(ts2, 'BTCUSD', 'SELL', 0.2, prices2['BTCUSD'])
    print(f"Sell executed: {sell_success}")
    print(f"After Trade - Cash: ${portfolio.current_cash:,.2f}, Positions: {portfolio.current_positions}")

    ts3 = dates[2]
    prices3 = pd.Series({'BTCUSD': market_prices['BTCUSD'].loc[ts3], 'ETHUSD': market_prices['ETHUSD'].loc[ts3]})
    portfolio.update_market_value(ts3, prices3)
    print(f"\n[{ts3}] Portfolio Value: ${portfolio.current_total_value:,.2f}")
    print(f"Cash: ${portfolio.current_cash:,.2f}, Positions: {portfolio.current_positions}")

    print("\nAttempting BUY ETHUSD...")
    buy_eth_success = portfolio.transact_asset(ts3, 'ETHUSD', 'BUY', 5, prices3['ETHUSD'])
    print(f"Buy ETH executed: {buy_eth_success}")
    print(f"After Trade - Cash: ${portfolio.current_cash:,.2f}, Positions: {portfolio.current_positions}")


    ts4 = dates[3]
    prices4 = pd.Series({'BTCUSD': market_prices['BTCUSD'].loc[ts4], 'ETHUSD': market_prices['ETHUSD'].loc[ts4]})
    portfolio.update_market_value(ts4, prices4)
    print(f"\n[{ts4}] Portfolio Value: ${portfolio.current_total_value:,.2f}")
    print(f"Cash: ${portfolio.current_cash:,.2f}, Positions: {portfolio.current_positions}")

    print("\nAttempting BUY BTCUSD (Insufficient Funds)...")
    buy_fail_success = portfolio.transact_asset(ts4, 'BTCUSD', 'BUY', 5.0, prices4['BTCUSD'])
    print(f"Buy executed: {buy_fail_success}")
    print(f"After Failed Trade - Cash: ${portfolio.current_cash:,.2f}, Positions: {portfolio.current_positions}")

    print("\n--- Simulation End ---")

    history_df = portfolio.get_history_df()
    print("\nPortfolio History:")
    print(history_df)

    trade_log_df = portfolio.get_trade_log_df()
    print("\nTrade Log:")
    print(trade_log_df)

    print("\nFinal Portfolio State:")
    print(f"Final Cash: ${portfolio.current_cash:,.2f}")
    print(f"Final Positions: {portfolio.current_positions}")
    print(f"Final Holdings Value: {portfolio.current_holdings_value}")
    print(f"Final Total Value: ${portfolio.current_total_value:,.2f}")
