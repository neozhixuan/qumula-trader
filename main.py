#!/usr/bin/env python3
"""
Fixed Multi-Agent Crypto Backtesting Framework
Fixes issues with price explosion and tiny trade amounts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Standardized signal format for all agents"""
    agent_id: str
    timestamp: datetime
    symbol: str
    direction: int  # -1 (sell), 0 (hold), 1 (buy)
    strength: float  # [0, 1] how strong the signal is
    confidence: float  # [0, 1] how confident the agent is
    features: Dict[str, float]  # Supporting data
    latency_ms: float  # How long this signal took to generate


@dataclass
class Trade:
    """Record of executed trade"""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    fees: float
    rationale: str  # Why this trade was made


@dataclass
class MarketData:
    """Single market data point"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class Agent(ABC):
    """Base class for all trading agents"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.last_signal_time = None
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, current_time: datetime) -> Signal:
        """Generate trading signal based on current market data"""
        pass


class QuantAgent(Agent):
    """Technical analysis agent using simple indicators"""
    
    def __init__(self, agent_id: str = "quant_agent"):
        super().__init__(agent_id)
        self.sma_short = 5
        self.sma_long = 20
        self.rsi_period = 14
    
    def calculate_sma(self, prices: pd.Series, period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return prices.mean()
        return prices.tail(period).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Handle division by zero
        loss = loss.replace(0, 0.0001)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def generate_signal(self, data: pd.DataFrame, current_time: datetime) -> Signal:
        """Generate signal based on SMA crossover and RSI"""
        start_time = time.time()
        
        if len(data) < self.sma_long:
            # Not enough data
            return Signal(
                agent_id=self.agent_id,
                timestamp=current_time,
                symbol=data['symbol'].iloc[-1] if len(data) > 0 else "BTC-USDT",
                direction=0,
                strength=0.0,
                confidence=0.0,
                features={},
                latency_ms=(time.time() - start_time) * 1000
            )
        
        prices = data['close']
        
        # Calculate indicators
        sma_short = self.calculate_sma(prices, self.sma_short)
        sma_long = self.calculate_sma(prices, self.sma_long)
        rsi = self.calculate_rsi(prices, self.rsi_period)
        current_price = prices.iloc[-1]
        
        # Signal logic - FIXED: More conservative thresholds
        direction = 0
        strength = 0.0
        confidence = 0.5
        
        # SMA crossover - FIXED: More reasonable thresholds
        if sma_long > 0:  # Avoid division by zero
            sma_ratio = sma_short / sma_long
            if sma_ratio > 1.005:  # Short MA 0.5% above long MA (was 2%)
                direction = 1
                strength = min((sma_ratio - 1.0) * 50, 0.8)  # Cap strength at 0.8
            elif sma_ratio < 0.995:  # Short MA 0.5% below long MA (was 2%)
                direction = -1
                strength = min((1.0 - sma_ratio) * 50, 0.8)  # Cap strength at 0.8
        
        # RSI confirmation - FIXED: More conservative bounds
        if direction == 1 and rsi < 65:  # Buy signal, not overbought (was 70)
            confidence = 0.7  # Reduced from 0.8
        elif direction == -1 and rsi > 35:  # Sell signal, not oversold (was 30)
            confidence = 0.7  # Reduced from 0.8
        elif direction == 1 and rsi > 75:  # Buy signal but overbought (was 80)
            confidence = 0.2  # Reduced from 0.3
        elif direction == -1 and rsi < 25:  # Sell signal but oversold (was 20)
            confidence = 0.2  # Reduced from 0.3
        
        features = {
            'sma_short': sma_short,
            'sma_long': sma_long,
            'sma_ratio': sma_ratio if sma_long > 0 else 1.0,
            'rsi': rsi,
            'current_price': current_price
        }
        
        latency_ms = (time.time() - start_time) * 1000
        
        return Signal(
            agent_id=self.agent_id,
            timestamp=current_time,
            symbol=data['symbol'].iloc[-1],
            direction=direction,
            strength=strength,
            confidence=confidence,
            features=features,
            latency_ms=latency_ms
        )


class SentimentAgent(Agent):
    """Simplified sentiment agent (mock implementation for now)"""
    
    def __init__(self, agent_id: str = "sentiment_agent"):
        super().__init__(agent_id)
        # In real implementation, this would connect to news/social media APIs
        self.sentiment_cache = {}
    
    def mock_sentiment_analysis(self, symbol: str, current_time: datetime) -> Tuple[float, float]:
        """Mock sentiment analysis - replace with real FinBERT in production"""
        # Simulate some time delay for sentiment processing
        time.sleep(0.001)  # FIXED: Reduced from 0.1s to 1ms to speed up backtest
        
        # Generate semi-random sentiment based on recent price action and time
        # In real implementation, this would analyze news/tweets
        seed = hash(f"{symbol}_{current_time.hour}") % 1000
        np.random.seed(seed)
        
        # FIXED: Much more conservative sentiment
        sentiment = np.random.normal(0, 0.1)  # Reduced volatility from 0.3 to 0.1
        confidence = np.random.uniform(0.3, 0.6)  # Reduced max confidence from 0.9 to 0.6
        
        return np.clip(sentiment, -1, 1), confidence
    
    def generate_signal(self, data: pd.DataFrame, current_time: datetime) -> Signal:
        """Generate signal based on sentiment analysis"""
        start_time = time.time()
        
        if len(data) == 0:
            return Signal(
                agent_id=self.agent_id,
                timestamp=current_time,
                symbol="BTC-USDT",
                direction=0,
                strength=0.0,
                confidence=0.0,
                features={},
                latency_ms=0
            )
        
        symbol = data['symbol'].iloc[-1]
        
        # Mock sentiment analysis
        sentiment_score, confidence = self.mock_sentiment_analysis(symbol, current_time)
        
        # Convert sentiment to trading signal - FIXED: Higher threshold
        direction = 0
        strength = 0.0
        
        if sentiment_score > 0.3:  # FIXED: Increased from 0.2
            direction = 1
            strength = min(sentiment_score * 0.5, 0.6)  # FIXED: Cap at 0.6 instead of 1.0
        elif sentiment_score < -0.3:  # FIXED: Increased from -0.2
            direction = -1
            strength = min(abs(sentiment_score) * 0.5, 0.6)  # FIXED: Cap at 0.6
        
        features = {
            'sentiment_score': sentiment_score,
            'news_volume': np.random.randint(10, 50),  # FIXED: Reduced range
            'social_mentions': np.random.randint(50, 200)  # FIXED: Reduced range
        }
        
        latency_ms = (time.time() - start_time) * 1000
        
        return Signal(
            agent_id=self.agent_id,
            timestamp=current_time,
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            features=features,
            latency_ms=latency_ms
        )


class Reconciler:
    """Combines signals from multiple agents"""
    
    def __init__(self, method: str = "weighted"):
        self.method = method
        self.agent_weights = {
            "quant_agent": 0.7,  # FIXED: Increased quant weight
            "sentiment_agent": 0.3  # FIXED: Decreased sentiment weight
        }
    
    def reconcile_signals(self, signals: List[Signal]) -> Optional[Signal]:
        """Combine multiple agent signals into single decision"""
        if not signals:
            return None
        
        start_time = time.time()
        
        if self.method == "weighted":
            return self._weighted_reconciliation(signals, start_time)
        elif self.method == "majority":
            return self._majority_reconciliation(signals, start_time)
        else:
            return self._confidence_reconciliation(signals, start_time)
    
    def _weighted_reconciliation(self, signals: List[Signal], start_time: float) -> Signal:
        """Confidence-weighted combination of signals"""
        total_weight = 0
        weighted_direction = 0
        weighted_strength = 0
        max_confidence = 0
        features_combined = {}
        
        for signal in signals:
            # Weight by both agent importance and signal confidence
            agent_weight = self.agent_weights.get(signal.agent_id, 0.5)
            final_weight = agent_weight * signal.confidence
            
            weighted_direction += signal.direction * signal.strength * final_weight
            weighted_strength += signal.strength * final_weight
            total_weight += final_weight
            max_confidence = max(max_confidence, signal.confidence)
            
            # Combine features
            for key, value in signal.features.items():
                features_combined[f"{signal.agent_id}_{key}"] = value
        
        if total_weight == 0:
            final_direction = 0
            final_strength = 0
        else:
            # FIXED: Higher threshold for action and cap strength
            final_direction = 1 if weighted_direction > 0.2 else (-1 if weighted_direction < -0.2 else 0)
            final_strength = min(abs(weighted_direction) / total_weight, 0.7)  # FIXED: Cap at 0.7
        
        # Combined confidence is average weighted by individual confidences
        final_confidence = min(max_confidence * (total_weight / len(signals)), 0.8)  # FIXED: Cap at 0.8
        
        return Signal(
            agent_id="reconciler",
            timestamp=signals[0].timestamp,
            symbol=signals[0].symbol,
            direction=final_direction,
            strength=final_strength,
            confidence=final_confidence,
            features=features_combined,
            latency_ms=(time.time() - start_time) * 1000  # FIXED: Proper timing
        )
    
    def _majority_reconciliation(self, signals: List[Signal], start_time: float) -> Signal:
        """Simple majority vote"""
        directions = [s.direction for s in signals]
        from collections import Counter
        direction_counts = Counter(directions)
        final_direction = direction_counts.most_common(1)[0][0]
        
        # Average strength and confidence of signals with winning direction
        matching_signals = [s for s in signals if s.direction == final_direction]
        final_strength = min(np.mean([s.strength for s in matching_signals]), 0.7)  # FIXED: Cap
        final_confidence = min(np.mean([s.confidence for s in matching_signals]), 0.8)  # FIXED: Cap
        
        return Signal(
            agent_id="reconciler",
            timestamp=signals[0].timestamp,
            symbol=signals[0].symbol,
            direction=final_direction,
            strength=final_strength,
            confidence=final_confidence,
            features={},
            latency_ms=(time.time() - start_time) * 1000
        )
    
    def _confidence_reconciliation(self, signals: List[Signal], start_time: float) -> Signal:
        """Take signal from most confident agent"""
        best_signal = max(signals, key=lambda s: s.confidence)
        return Signal(
            agent_id="reconciler",
            timestamp=best_signal.timestamp,
            symbol=best_signal.symbol,
            direction=best_signal.direction,
            strength=min(best_signal.strength, 0.7),  # FIXED: Cap strength
            confidence=min(best_signal.confidence, 0.8),  # FIXED: Cap confidence
            features=best_signal.features,
            latency_ms=(time.time() - start_time) * 1000
        )


class ExecutorAgent:
    """Decides position sizing and executes trades"""
    
    def __init__(self, initial_balance: float = 10000):
        self.balance = initial_balance
        self.positions = {}  # symbol -> quantity
        self.max_position_pct = 0.05  # FIXED: Reduced from 10% to 5% per position
        self.transaction_fee = 0.001  # 0.1% fee
        self.trades = []
        self.min_trade_value = 50  # FIXED: Minimum trade value to avoid dust trades
    
    def execute_signal(self, signal: Signal, current_price: float) -> Optional[Trade]:
        """Convert signal to actual trade"""
        # FIXED: Higher threshold for signal strength
        if signal.direction == 0 or signal.strength < 0.4 or signal.confidence < 0.5:
            return None  # No trade if signal too weak
        
        symbol = signal.symbol
        current_position = self.positions.get(symbol, 0)
        
        # FIXED: More conservative position sizing
        risk_per_trade = 0.01  # FIXED: Reduced from 2% to 1% of portfolio at risk
        base_position_value = self.balance * risk_per_trade
        
        # Scale by signal strength and confidence (but cap the multiplier)
        signal_multiplier = min(signal.strength * signal.confidence, 0.5)  # FIXED: Cap multiplier
        position_value = base_position_value * signal_multiplier
        
        # FIXED: Ensure minimum trade value
        if position_value < self.min_trade_value:
            return None  # Skip dust trades
        
        quantity = position_value / current_price
        
        # Apply position limits
        max_position_value = self.balance * self.max_position_pct
        max_quantity = max_position_value / current_price
        quantity = min(quantity, max_quantity)
        
        if signal.direction == 1:  # Buy
            action = "BUY"
            required_cash = quantity * current_price * (1 + self.transaction_fee)
            if required_cash > self.balance * 0.95:  # FIXED: Leave 5% cash buffer
                # Scale down the trade to fit available cash
                max_affordable = (self.balance * 0.95) / (current_price * (1 + self.transaction_fee))
                quantity = min(quantity, max_affordable)
                if quantity * current_price < self.min_trade_value:
                    return None  # Skip if still too small
        else:  # Sell
            action = "SELL"
            if current_position <= 0:
                return None  # Nothing to sell
            quantity = min(quantity, abs(current_position))
            if quantity * current_price < self.min_trade_value:
                return None  # Skip dust trades
        
        # Calculate fees
        trade_value = quantity * current_price
        fees = trade_value * self.transaction_fee
        
        # FIXED: Add validation before executing
        if action == "BUY" and (trade_value + fees) > self.balance:
            return None  # Skip if insufficient funds
        
        # Execute trade
        if action == "BUY":
            self.balance -= (trade_value + fees)
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:  # SELL
            self.balance += (trade_value - fees)
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        
        # Create trade record
        trade = Trade(
            timestamp=signal.timestamp,
            symbol=symbol,
            action=action,
            price=current_price,
            quantity=quantity,
            fees=fees,
            rationale=f"Agent: {signal.agent_id}, Strength: {signal.strength:.2f}, Confidence: {signal.confidence:.2f}"
        )
        
        self.trades.append(trade)
        return trade
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        total_value = self.balance
        for symbol, quantity in self.positions.items():
            if symbol in prices and quantity > 0:
                total_value += quantity * prices[symbol]
        return total_value


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, agents: List[Agent], reconciler: Reconciler, executor: ExecutorAgent):
        self.agents = agents
        self.reconciler = reconciler
        self.executor = executor
        self.results = []
        self.latency_stats = []
    
    def run_backtest(self, data: pd.DataFrame, lookback_window: int = 50) -> Dict:
        """Run backtest on historical data"""
        logger.info(f"Starting backtest with {len(data)} data points")
        
        portfolio_values = []
        daily_returns = []
        
        # Ensure data is sorted by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # FIXED: Add progress tracking and limit iterations for testing
        max_iterations = min(len(data), 2000)  # Limit for faster testing
        
        for i in range(lookback_window, max_iterations):
            if i % 100 == 0:  # Progress update every 100 iterations
                logger.info(f"Processing data point {i}/{max_iterations}")
            
            current_time = data.loc[i, 'timestamp']
            current_data = data.loc[max(0, i-lookback_window):i].copy()
            current_price = data.loc[i, 'close']
            
            # FIXED: Sanity check on price
            if current_price <= 0 or current_price > 1000000:
                logger.warning(f"Abnormal price detected: ${current_price:.2f} at {current_time}")
                continue
            
            # Get signals from all agents
            signals = []
            total_latency = 0
            
            for agent in self.agents:
                try:
                    signal = agent.generate_signal(current_data, current_time)
                    signals.append(signal)
                    total_latency += signal.latency_ms
                except Exception as e:
                    logger.warning(f"Agent {agent.agent_id} failed: {e}")
                    continue
            
            # Reconcile signals
            if signals:
                try:
                    final_signal = self.reconciler.reconcile_signals(signals)
                    if final_signal:
                        # Debug logging for reconciled signals
                        if i < 60 and final_signal.direction != 0:
                            logger.info(f"Final signal: direction={final_signal.direction}, strength={final_signal.strength:.3f}, confidence={final_signal.confidence:.3f}")
                        
                        # Execute trade
                        trade = self.executor.execute_signal(final_signal, current_price)
                        if trade:
                            logger.info(f"Executed {trade.action} {trade.quantity:.6f} {trade.symbol} at ${trade.price:.2f}")
                        elif i < 60 and final_signal.direction != 0:
                            logger.info(f"Signal rejected by executor: strength={final_signal.strength:.3f}, confidence={final_signal.confidence:.3f}")
                except Exception as e:
                    logger.warning(f"Signal reconciliation failed: {e}")
            
            # Calculate portfolio value
            current_prices = {data.loc[i, 'symbol']: current_price}
            portfolio_value = self.executor.get_portfolio_value(current_prices)
            portfolio_values.append(portfolio_value)
            
            # FIXED: Sanity check on portfolio value
            if portfolio_value <= 0:
                logger.error(f"Portfolio value became non-positive: {portfolio_value}")
                break
            
            # Store latency stats
            self.latency_stats.append({
                'timestamp': current_time,
                'total_latency_ms': total_latency,
                'num_agents': len(signals)
            })
            
            # Calculate daily return
            if len(portfolio_values) > 1:
                daily_return = (portfolio_value / portfolio_values[-2]) - 1
                # FIXED: Cap extreme returns to detect issues
                if abs(daily_return) > 0.5:  # 50% return in one period seems extreme
                    logger.warning(f"Extreme return detected: {daily_return:.2%} at {current_time}")
                daily_returns.append(daily_return)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(portfolio_values, daily_returns, data)
        self.results = results
        
        logger.info("Backtest completed")
        return results
    
    def _calculate_performance_metrics(self, portfolio_values: List[float], 
                                     daily_returns: List[float], data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance statistics"""
        if not portfolio_values or not daily_returns:
            return {}
        
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value / initial_value) - 1
        
        returns_array = np.array(daily_returns)
        
        # Basic metrics
        avg_return = np.mean(returns_array)
        volatility = np.std(returns_array)
        sharpe_ratio = (avg_return / volatility * np.sqrt(252)) if volatility > 0 else 0
        
        # Drawdown calculation
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) / running_max) - 1
        max_drawdown = np.min(drawdown)
        
        # Trade statistics
        num_trades = len(self.executor.trades)
        
        # Buy and hold benchmark
        start_price = data.iloc[0]['close']
        end_price = data.iloc[min(len(data)-1, 1999)]['close']  # FIXED: Handle limited data
        buy_hold_return = (end_price / start_price) - 1
        
        # Latency statistics
        if self.latency_stats:
            avg_latency = np.mean([stat['total_latency_ms'] for stat in self.latency_stats])
            max_latency = np.max([stat['total_latency_ms'] for stat in self.latency_stats])
        else:
            avg_latency = max_latency = 0
        
        results = {
            'initial_balance': initial_value,
            'final_balance': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility * np.sqrt(252),  # Annualized
            'num_trades': num_trades,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns,
            'trades': self.executor.trades
        }
        
        return results
    
    def print_results(self):
        """Print formatted backtest results"""
        if not self.results:
            print("No results to display. Run backtest first.")
            return
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Initial Balance: ${self.results['initial_balance']:,.2f}")
        print(f"Final Balance: ${self.results['final_balance']:,.2f}")
        print(f"Total Return: {self.results['total_return']:.2%}")
        print(f"Buy & Hold Return: {self.results['buy_hold_return']:.2%}")
        print(f"Excess Return: {self.results['excess_return']:.2%}")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {self.results['max_drawdown']:.2%}")
        print(f"Volatility (Annual): {self.results['volatility']:.2%}")
        print(f"Number of Trades: {self.results['num_trades']}")
        print(f"Average Latency: {self.results['avg_latency_ms']:.1f}ms")
        print(f"Max Latency: {self.results['max_latency_ms']:.1f}ms")
        print("="*50)


def generate_sample_data(symbol: str = "BTC-USDT", days: int = 90) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results
    
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start_date, periods=days*24, freq='1h')  # FIXED: Hourly instead of minute data
    
    # FIXED: Much more realistic price generation
    initial_price = 45000  # Starting BTC price
    
    # Generate realistic returns with much lower volatility
    base_returns = np.random.normal(0.0001, 0.005, len(dates))  # FIXED: Reduced volatility significantly
    
    # Add some structure but keep it reasonable
    for i in range(len(base_returns)):
        # Small weekly cycle
        weekly_effect = 0.0005 * np.sin(2 * np.pi * i / (7 * 24))  # FIXED: Reduced amplitude
        # Very small momentum effect
        if i > 24:  # Look back 24 hours
            momentum = np.mean(base_returns[i-24:i]) * 0.05  # FIXED: Much smaller momentum
        else:
            momentum = 0
        
        base_returns[i] += weekly_effect + momentum
        
        # FIXED: Cap extreme returns
        base_returns[i] = np.clip(base_returns[i], -0.05, 0.05)  # Max 5% move per hour
    
    # Generate price series
    prices = [initial_price]
    for i in range(1, len(base_returns)):
        new_price = prices[-1] * (1 + base_returns[i])
        # FIXED: Additional safeguard against price explosion
        if new_price > prices[-1] * 1.1:  # Max 10% jump
            new_price = prices[-1] * 1.1
        elif new_price < prices[-1] * 0.9:  # Max 10% drop
            new_price = prices[-1] * 0.9
        prices.append(new_price)
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(dates, prices)):
        # FIXED: Much more realistic OHLCV generation
        daily_volatility = abs(base_returns[i]) * close * 0.5  # FIXED: Reduced volatility
        
        high = close + np.random.uniform(0, daily_volatility * 0.5)
        low = close - np.random.uniform(0, daily_volatility * 0.5)
        
        # Ensure reasonable open price
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] + np.random.uniform(-daily_volatility*0.2, daily_volatility*0.2)
        
        # Ensure OHLC relationships are valid
        high = max(open_price, high, close)
        low = min(open_price, low, close)
        
        volume = np.random.uniform(500, 2000)  # FIXED: More reasonable volume range
        
        data.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # FIXED: Final validation
    logger.info(f"Generated data - Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    logger.info(f"Max price change: {((df['close'].max() / df['close'].min()) - 1):.2%}")
    
    return df


def main():
    """Example usage of the backtesting framework"""
    print("Multi-Agent Crypto Trading Backtest")
    print("Generating sample data...")
    
    # FIXED: Generate more reasonable dataset
    data = generate_sample_data("BTC-USDT", days=30)  # 30 days of hourly data
    print(f"Generated {len(data)} data points from {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Initialize agents
    quant_agent = QuantAgent("quant_agent")
    sentiment_agent = SentimentAgent("sentiment_agent")
    agents = [quant_agent, sentiment_agent]
    
    # Initialize reconciler and executor
    reconciler = Reconciler(method="weighted")
    executor = ExecutorAgent(initial_balance=10000)
    
    # Run backtest
    engine = BacktestEngine(agents, reconciler, executor)
    
    print("Running backtest...")
    results = engine.run_backtest(data, lookback_window=48)  # Use 48 hours of lookback
    
    if not results:
        print("Backtest failed!")
        return
    
    # Print results
    engine.print_results()
    
    # Show some sample trades
    if results.get('trades'):
        print(f"\nSample trades:")
        for i, trade in enumerate(results['trades'][:10]):  # Show up to 10 trades
            print(f"  {trade.timestamp.strftime('%Y-%m-%d %H:%M')}: {trade.action} {trade.quantity:.6f} {trade.symbol} at ${trade.price:.2f} (Value: ${trade.quantity * trade.price:.2f})")
        
        if len(results['trades']) > 10:
            print(f"  ... and {len(results['trades']) - 10} more trades")
            
        # Show trade value distribution
        trade_values = [t.quantity * t.price for t in results['trades']]
        if trade_values:
            print(f"\nTrade value stats:")
            print(f"  Average trade value: ${np.mean(trade_values):.2f}")
            print(f"  Min trade value: ${np.min(trade_values):.2f}")
            print(f"  Max trade value: ${np.max(trade_values):.2f}")
    else:
        print("\nNo trades executed during backtest period")
    
    print(f"\nBacktest completed successfully!")
    print(f"Final portfolio balance: ${executor.balance:.2f}")
    
    # Show current positions
    if executor.positions:
        print("Current positions:")
        total_position_value = 0
        final_price = data['close'].iloc[-1]
        for symbol, quantity in executor.positions.items():
            if quantity > 0.000001:  # Show positions above dust level
                position_value = quantity * final_price
                total_position_value += position_value
                print(f"  {symbol}: {quantity:.6f} (${position_value:.2f})")
        print(f"Total position value: ${total_position_value:.2f}")
        print(f"Total portfolio value: ${executor.balance + total_position_value:.2f}")
    else:
        print("No open positions")
    
    # Additional diagnostics
    print(f"\nDiagnostics:")
    print(f"  Data points processed: {len(results.get('portfolio_values', []))}")
    print(f"  Final BTC price: ${data['close'].iloc[-1]:.2f}")
    print(f"  Price change: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1):.2%}")


if __name__ == "__main__":
    main() 