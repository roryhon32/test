import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from binance.client import Client
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta
import warnings
import json
import pickle
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
import math
from typing import Dict, List, Tuple, Optional
import logging

warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Análise de sentiment de redes sociais"""
    
    def __init__(self):
        self.fear_greed_url = "https://api.alternative.me/fng/"
        
    def get_fear_greed_index(self):
        """Pega o índice Fear & Greed"""
        try:
            response = requests.get(self.fear_greed_url)
            data = response.json()
            return float(data['data'][0]['value'])
        except:
            return 50.0  # Neutro se falhar
    


class AdvancedDataCollector:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
        self.scalers = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def get_multiple_timeframes(self, symbol, timeframes=['1h', '4h', '1d'], days=60):
        """Coleta dados de múltiplos timeframes"""
        all_data = {}
        
        for tf in timeframes:
            try:
                klines = self.client.get_historical_klines(
                    symbol, tf, f"{days} days ago UTC"
                )
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'number_of_trades']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                all_data[tf] = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'number_of_trades']]
                
            except Exception as e:
                logger.error(f"Erro ao coletar dados {tf}: {e}")
                
        return all_data
    
    def get_orderbook_data(self, symbol, limit=100):
        """Coleta dados do order book"""
        try:
            orderbook = self.client.get_order_book(symbol=symbol, limit=limit)
            
            bids = np.array([[float(bid[0]), float(bid[1])] for bid in orderbook['bids'][:20]])
            asks = np.array([[float(ask[0]), float(ask[1])] for ask in orderbook['asks'][:20]])
            
            # Métricas do order book
            bid_ask_spread = asks[0][0] - bids[0][0]
            bid_volume = np.sum(bids[:, 1])
            ask_volume = np.sum(asks[:, 1])
            volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            return {
                'spread': bid_ask_spread,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'volume_imbalance': volume_imbalance,
                'depth_ratio': bid_volume / ask_volume if ask_volume > 0 else 1
            }
        except:
            return {'spread': 0, 'bid_volume': 0, 'ask_volume': 0, 'volume_imbalance': 0, 'depth_ratio': 1}
    
    def add_advanced_indicators(self, df, timeframe='1h'):
        """Adiciona indicadores técnicos avançados"""
        df = df.copy()
        
        # Indicadores básicos
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Médias móveis múltiplas
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator()
            df[f'ema_{period}'] = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()
        
        # ATR e volatilidade
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['volatility'] = df['close'].rolling(20).std()
        
        # Indicadores de volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ema'] = df['volume'].ewm(span=20, adjust=False).mean()
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        
        # Padrões de candlestick
        df['doji'] = self.detect_doji(df)
        df['hammer'] = self.detect_hammer(df)
        df['shooting_star'] = self.detect_shooting_star(df)
        
        # Fractais e suporte/resistência
        df['resistance'] = self.find_resistance_levels(df)
        df['support'] = self.find_support_levels(df)
        
        # Divergências
        df['rsi_divergence'] = self.detect_rsi_divergence(df)
        
        # Mudanças de preço em diferentes períodos
        for period in [1, 3, 5, 10, 20]:
            df[f'price_change_{period}'] = df['close'].pct_change(period)
            df[f'volume_change_{period}'] = df['volume'].pct_change(period)
        
        # Correlação com outros ativos (simulado)
        df['btc_correlation'] = np.random.normal(0.7, 0.1, len(df))  # Simular correlação com BTC
        
        return df.dropna()
    
    def detect_doji(self, df):
        """Detecta padrão Doji"""
        body = abs(df['close'] - df['open'])
        range_val = df['high'] - df['low']
        return (body / range_val < 0.1).astype(int)
    
    def detect_hammer(self, df):
        """Detecta padrão Hammer"""
        body = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        return ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int)
    
    def detect_shooting_star(self, df):
        """Detecta padrão Shooting Star"""
        body = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        return ((upper_shadow > 2 * body) & (lower_shadow < body)).astype(int)
    
    def find_resistance_levels(self, df, window=20):
        """Encontra níveis de resistência"""
        highs = df['high'].rolling(window).max()
        return highs
    
    def find_support_levels(self, df, window=20):
        """Encontra níveis de suporte"""
        lows = df['low'].rolling(window).min()
        return lows
    
    def detect_rsi_divergence(self, df, window=14):
        """Detecta divergências no RSI"""
        price_peaks = df['close'].rolling(window).max()
        rsi_peaks = df['rsi'].rolling(window).max()
        
        price_trend = price_peaks.diff()
        rsi_trend = rsi_peaks.diff()
        
        # Divergência bearish: preço sobe, RSI desce
        bearish_div = (price_trend > 0) & (rsi_trend < 0)
        # Divergência bullish: preço desce, RSI sobe
        bullish_div = (price_trend < 0) & (rsi_trend > 0)
        
        return (bullish_div.astype(int) - bearish_div.astype(int))

class TransformerModel(nn.Module):
    """Modelo Transformer para análise temporal"""
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self.create_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Buy/Hold/Sell
        )
    
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Projeção de entrada
        x = self.input_projection(x)
        
        # Adicionar encoding posicional
        if seq_len <= self.positional_encoding.size(1):
            x += self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer
        x = self.transformer(x)
        
        # Usar apenas o último token
        x = x[:, -1, :]
        
        return self.classifier(x)

class EnsembleModel(nn.Module):
    """Ensemble de múltiplos modelos"""
    
    def __init__(self, input_size, sequence_length):
        super(EnsembleModel, self).__init__()
        
        # LSTM Model
        self.lstm = nn.LSTM(input_size, 128, 2, batch_first=True, dropout=0.2)
        self.lstm_classifier = nn.Linear(128, 3)
        
        # GRU Model
        self.gru = nn.GRU(input_size, 128, 2, batch_first=True, dropout=0.2)
        self.gru_classifier = nn.Linear(128, 3)
        
        # CNN Model
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.cnn_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 3)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(input_size, num_heads=8, batch_first=True)
        self.attention_classifier = nn.Linear(input_size, 3)
        
        # Final ensemble layer
        self.ensemble_classifier = nn.Sequential(
            nn.Linear(12, 64),  # 4 models * 3 classes each
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # LSTM branch
        lstm_out, _ = self.lstm(x)
        lstm_pred = self.lstm_classifier(lstm_out[:, -1, :])
        
        # GRU branch
        gru_out, _ = self.gru(x)
        gru_pred = self.gru_classifier(gru_out[:, -1, :])
        
        # CNN branch
        x_cnn = x.transpose(1, 2)  # (batch, features, seq_len)
        cnn_out = torch.relu(self.conv1(x_cnn))
        cnn_out = torch.relu(self.conv2(cnn_out))
        cnn_pred = self.cnn_classifier(cnn_out)
        
        # Attention branch
        attn_out, _ = self.attention(x, x, x)
        attn_pred = self.attention_classifier(attn_out[:, -1, :])
        
        # Combine predictions
        ensemble_input = torch.cat([lstm_pred, gru_pred, cnn_pred, attn_pred], dim=1)
        final_pred = self.ensemble_classifier(ensemble_input)
        
        return final_pred

class RLTradingAgent:
    """Agente de Reinforcement Learning para trading"""
    
    def __init__(self, state_size, action_size=3, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        
        # Rede neural para Q-learning
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
    
    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class RiskManager:
    """Gerenciamento de risco avançado"""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_drawdown = 0.15  # 15% máximo de drawdown
        self.position_size_limit = 0.1  # 10% do portfolio por posição
        self.stop_loss = 0.02  # 2% stop loss
        self.take_profit = 0.06  # 6% take profit
        self.var_confidence = 0.05  # 5% VaR
        
    def calculate_position_size(self, balance, price, volatility, confidence):
        """Calcula tamanho da posição baseado no Kelly Criterion modificado"""
        
        # Kelly Criterion: f = (bp - q) / b
        # onde b = odds, p = prob. de ganho, q = prob. de perda
        win_prob = max(0.5, confidence)
        loss_prob = 1 - win_prob
        
        # Ratio risco/recompensa
        risk_reward_ratio = self.take_profit / self.stop_loss
        
        # Kelly fraction
        kelly_fraction = (win_prob * risk_reward_ratio - loss_prob) / risk_reward_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Limitado a 25%
        
        # Ajustar pela volatilidade
        volatility_adjustment = 1 / (1 + volatility)
        
        # Tamanho final da posição
        position_fraction = kelly_fraction * volatility_adjustment * self.position_size_limit
        position_value = balance * position_fraction
        
        return position_value / price
    
    def calculate_var(self, returns, confidence_level=0.05):
        """Calcula Value at Risk"""
        if len(returns) < 30:
            return 0
        
        sorted_returns = np.sort(returns)
        var_index = int(confidence_level * len(sorted_returns))
        return abs(sorted_returns[var_index])
    
    def should_stop_trading(self, current_balance):
        """Verifica se deve parar de tradear por drawdown excessivo"""
        drawdown = (self.initial_balance - current_balance) / self.initial_balance
        return drawdown > self.max_drawdown

class AdvancedBacktester:
    """Sistema de backtesting avançado"""
    
    def __init__(self, initial_balance=10000, commission=0.001):
        self.initial_balance = initial_balance
        self.commission = commission
        self.trades = []
        self.equity_curve = []
        self.risk_manager = RiskManager(initial_balance)
    
    def run_backtest(self, df, predictions, probabilities):
        """Executa backtest completo"""
        
        balance = self.initial_balance
        position = 0
        entry_price = 0
        entry_time = None
        
        for i in range(len(predictions)):
            current_price = df.iloc[i]['close']
            current_time = df.iloc[i]['timestamp']
            predicted_action = predictions[i]
            confidence = probabilities[i]
            
            # Calcular volatilidade
            if i >= 20:
                volatility = df['close'].iloc[i-20:i].pct_change().std()
            else:
                volatility = 0.02
            
            # Gerenciamento de posição existente
            if position != 0:
                # Check stop loss
                if position > 0 and (current_price - entry_price) / entry_price <= -self.risk_manager.stop_loss:
                    balance = self.close_position(balance, position, current_price, entry_price, current_time, 'stop_loss')
                    position = 0
                    
                # Check take profit
                elif position > 0 and (current_price - entry_price) / entry_price >= self.risk_manager.take_profit:
                    balance = self.close_position(balance, position, current_price, entry_price, current_time, 'take_profit')
                    position = 0
                    
                # Check opposite signal
                elif (position > 0 and predicted_action == 2) or (position < 0 and predicted_action == 1):
                    balance = self.close_position(balance, position, current_price, entry_price, current_time, 'signal')
                    position = 0
            
            # Abrir nova posição
            if position == 0 and predicted_action != 0 and confidence > 0.6:
                position_size = self.risk_manager.calculate_position_size(
                    balance, current_price, volatility, confidence
                )
                
                if predicted_action == 1:  # Buy
                    position = position_size
                    entry_price = current_price
                    entry_time = current_time
                    balance -= position_size * current_price * (1 + self.commission)
                    
                elif predicted_action == 2:  # Sell (short)
                    position = -position_size
                    entry_price = current_price
                    entry_time = current_time
                    balance += position_size * current_price * (1 - self.commission)
            
            # Calcular equity atual
            current_equity = balance
            if position != 0:
                if position > 0:
                    current_equity += position * current_price
                else:
                    current_equity += position * (2 * entry_price - current_price)
            
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': current_equity,
                'balance': balance,
                'position': position,
                'price': current_price
            })
        
        return self.calculate_metrics()
    
    def close_position(self, balance, position, current_price, entry_price, current_time, reason):
        """Fecha posição e registra trade"""
        
        if position > 0:  # Long position
            pnl = position * (current_price - entry_price)
            balance += position * current_price * (1 - self.commission)
        else:  # Short position
            pnl = -position * (current_price - entry_price)
            balance += -position * current_price * (1 + self.commission)
        
        trade_record = {
            'entry_price': entry_price,
            'exit_price': current_price,
            'position_size': abs(position),
            'direction': 'long' if position > 0 else 'short',
            'pnl': pnl,
            'return': pnl / (abs(position) * entry_price),
            'exit_time': current_time,
            'reason': reason
        }
        
        self.trades.append(trade_record)
        return balance
    
    def calculate_metrics(self):
        """Calcula métricas de performance"""
        
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Métricas básicas
        total_return = (equity_df['equity'].iloc[-1] - self.initial_balance) / self.initial_balance
        
        # Sharpe Ratio
        equity_returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = equity_returns.mean() / equity_returns.std() * np.sqrt(252) if equity_returns.std() > 0 else 0
        
        # Maximum Drawdown
        running_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win Rate
        if len(trades_df) > 0:
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades_df),
            'final_equity': equity_df['equity'].iloc[-1]
        }

class AdvancedTradingSystem:
    """Sistema de trading completo e avançado"""
    
    def __init__(self, api_key, api_secret):
        self.data_collector = AdvancedDataCollector(api_key, api_secret)
        self.models = {}
        self.scalers = {}
        self.sequence_length = 60
        self.rl_agent = None
        self.risk_manager = RiskManager()
        
    def prepare_multi_timeframe_data(self, symbol, days=90):
        """Prepara dados de múltiplos timeframes"""
        
        # Coletar dados
        all_data = self.data_collector.get_multiple_timeframes(
            symbol, ['1h', '4h', '1d'], days
        )
        
        enhanced_data = {}
        
        for tf, df in all_data.items():
            # Adicionar indicadores técnicos
            df_enhanced = self.data_collector.add_advanced_indicators(df, tf)
            
            # Adicionar dados de sentiment (simulado)

            
           
  
            
            # Adicionar Fear & Greed Index
            df_enhanced['fear_greed'] = self.data_collector.sentiment_analyzer.get_fear_greed_index()
            
            enhanced_data[tf] = df_enhanced
        
        return enhanced_data
    
    def create_ensemble_features(self, multi_tf_data):
        """Cria features combinadas de múltiplos timeframes"""
        

        base_df = multi_tf_data['1h'].copy()
        base_df.set_index('timestamp', inplace=True)
        
        # Adicionar features de timeframes maiores
        for tf in ['4h', '1d']:
            if tf in multi_tf_data:
                tf_data = multi_tf_data[tf].copy()
                tf_data.set_index('timestamp', inplace=True)
        
                tf_resampled = tf_data.resample('1H').ffill()
                tf_resampled.columns = [f'{tf}_{col}' for col in tf_resampled.columns]
        
                base_df = base_df.join(tf_resampled, how='left')
        
        base_df.reset_index(inplace=True)
        base_df = base_df.fillna(method='ffill').dropna()
        
        return base_df

                
                # Merge com dados base
 

        
        # Remover colunas desnecessárias e preencher NaN
        base_df = base_df.fillna(method='ffill').dropna()
        
        return base_df

    def create_ensemble_features(self, multi_tf_data):
        """Cria features combinadas de múltiplos timeframes"""

        base_df = multi_tf_data['1h'].copy()

        # Garante que 'timestamp' esteja no índice antes de mesclar
        if 'timestamp' in base_df.columns:
            base_df.set_index('timestamp', inplace=True)

        # Adicionar features de timeframes maiores
        for tf in ['4h', '1d']:
            if tf in multi_tf_data:
                tf_data = multi_tf_data[tf].copy()

                # Garante que o timestamp está como índice
                if 'timestamp' in tf_data.columns:
                    tf_data.set_index('timestamp', inplace=True)

                # Reamostra para 1H e aplica preenchimento
                tf_resampled = tf_data.resample('1H').ffill()

                # Adiciona prefixo para evitar conflito de nomes
                tf_resampled.columns = [
                    f'{tf}_{col}' for col in tf_resampled.columns]

                # Junta com base_df
                base_df = base_df.join(tf_resampled, how='left')

        # Volta com timestamp como coluna
        base_df.reset_index(inplace=True)

        # Preenche valores faltantes
        base_df = base_df.fillna(method='ffill').dropna()

        return base_df

    def create_advanced_labels(self, df, method='multi_target'):
        """Cria labels avançadas para treinamento"""
        
        if method == 'multi_target':
            # Labels para diferentes horizontes de tempo
            labels = {}
            
            for horizon in [1, 3, 6, 12, 24]:  # 1h, 3h, 6h, 12h, 24h
                future_returns = df['close'].shift(-horizon) / df['close'] - 1
                
                # Classificação baseada em thresholds adaptativos
                volatility = df['close'].pct_change().rolling(20).std()
                
                labels[f'target_{horizon}h'] = np.where(
                    future_returns > 2 * volatility, 1,  # Buy
                    np.where(future_returns < -2 * volatility, 2, 0)  # Sell, Hold
                )
            
            return labels
        
        elif method == 'regime_based':
            # Labels baseadas em regime de mercado
            volatility = df['close'].pct_change().rolling(20).std()
            trend = df['close'].rolling(20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
            
            # Regimes: Trending Bull, Trending Bear, Ranging
            regime = np.where(
                (trend > 0) & (volatility < volatility.median()), 1,  # Trending Bull
                np.where(
                    (trend < 0) & (volatility < volatility.median()), 2,  # Trending Bear
                    0  # Ranging
                )
            )
            
            return {'regime': regime}
    
    def train_ensemble_model(self, X_train, y_train, X_val, y_val):
        """Treina ensemble de modelos"""
        
        input_size = X_train.shape[2]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Modelo Ensemble principal
        ensemble_model = EnsembleModel(input_size, self.sequence_length).to(device)
        
        # Modelo Transformer
        transformer_model = TransformerModel(input_size).to(device)
        
        # Random Forest para comparação
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        # Treinar modelos
        models = {
            'ensemble': self.train_pytorch_model(ensemble_model, X_train, y_train, X_val, y_val),
            'transformer': self.train_pytorch_model(transformer_model, X_train, y_train, X_val, y_val),
            'random_forest': self.train_sklearn_model(rf_model, X_train, y_train, X_val, y_val)
        }
        
        return models
    
    def train_pytorch_model(self, model, X_train, y_train, X_val, y_val, epochs=100):
        """Treina modelo PyTorch"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        criterion = nn.CrossEntropyLoss(weight=self.calculate_class_weights(y_train))
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for i in range(0, len(X_train), 32):
                batch_X = torch.FloatTensor(X_train[i:i+32]).to(device)
                batch_y = torch.LongTensor(y_train[i:i+32]).to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val), 32):
                    batch_X = torch.FloatTensor(X_val[i:i+32]).to(device)
                    batch_y = torch.LongTensor(y_val[i:i+32]).to(device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            scheduler.step()
            
            avg_train_loss = train_loss / (len(X_train) // 32 + 1)
            avg_val_loss = val_loss / (len(X_val) // 32 + 1)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, '
                           f'Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                           f'Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping at epoch {epoch}')
                    break
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pth'))
        return model
    
    def train_sklearn_model(self, model, X_train, y_train, X_val, y_val):
        """Treina modelo sklearn"""
        
        # Reshape para 2D
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_val_2d = X_val.reshape(X_val.shape[0], -1)
        
        # Treinar
        model.fit(X_train_2d, y_train)
        
        # Avaliar
        train_score = model.score(X_train_2d, y_train)
        val_score = model.score(X_val_2d, y_val)
        
        logger.info(f'Random Forest - Train Acc: {train_score:.3f}, Val Acc: {val_score:.3f}')
        
        return model
    
    def calculate_class_weights(self, y):
        """Calcula pesos para classes desbalanceadas"""
        unique, counts = np.unique(y, return_counts=True)
        weights = len(y) / (len(unique) * counts)
        weight_dict = dict(zip(unique, weights))
        
        # Converter para tensor
        weight_tensor = torch.FloatTensor([weight_dict.get(i, 1.0) for i in range(3)])
        
        return weight_tensor
    
    def train_reinforcement_learning(self, env_data, episodes=1000):
        """Treina agente de RL"""
        
        state_size = env_data.shape[1]
        self.rl_agent = RLTradingAgent(state_size)
        
        for episode in range(episodes):
            state = self.reset_environment(env_data)
            total_reward = 0
            
            for step in range(len(env_data) - 1):
                action = self.rl_agent.act(state)
                next_state, reward, done = self.step_environment(env_data, step, action)
                
                self.rl_agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            self.rl_agent.replay()
            
            if episode % 100 == 0:
                logger.info(f'RL Episode {episode}: Total Reward: {total_reward:.2f}')
        
        return self.rl_agent
    
    def reset_environment(self, data):
        """Reset do ambiente de RL"""
        return data[0]
    
    def step_environment(self, data, step, action):
        """Step do ambiente de RL"""
        current_price = data[step, 0]  # Assumindo que preço é a primeira feature
        next_price = data[step + 1, 0]
        
        # Calcular reward baseado na ação
        price_change = (next_price - current_price) / current_price
        
        if action == 1:  # Buy
            reward = price_change * 100
        elif action == 2:  # Sell
            reward = -price_change * 100
        else:  # Hold
            reward = 0
        
        next_state = data[step + 1]
        done = step >= len(data) - 2
        
        return next_state, reward, done
    
    def create_meta_features(self, predictions_dict):
        """Cria meta-features baseadas nas predições dos modelos"""
        
        meta_features = []
        
        for i in range(len(predictions_dict['ensemble'])):
            features = [
                predictions_dict['ensemble'][i],
                predictions_dict['transformer'][i],
                predictions_dict['random_forest'][i],
                
                # Concordância entre modelos
                np.std([predictions_dict['ensemble'][i], 
                       predictions_dict['transformer'][i], 
                       predictions_dict['random_forest'][i]]),
                
                # Confiança média
                np.mean([predictions_dict['ensemble'][i], 
                        predictions_dict['transformer'][i], 
                        predictions_dict['random_forest'][i]]),
            ]
            
            meta_features.append(features)
        
        return np.array(meta_features)
    
    def run_comprehensive_backtest(self, df, symbol):
        """Executa backtest completo com todos os modelos"""
        
        logger.info(f"Iniciando backtest completo para {symbol}")
        
        # Preparar dados
        enhanced_data = self.prepare_multi_timeframe_data(symbol)
        combined_df = self.create_ensemble_features(enhanced_data)
        
        # Preparar features
        feature_columns = [col for col in combined_df.columns if col not in ['timestamp']]
        features = combined_df[feature_columns].values
        
        # Normalizar features
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Criar sequências
        X, _ = self.create_sequences(features_scaled)
        
        # Criar labels
        labels_dict = self.create_advanced_labels(combined_df)
        y = labels_dict['target_1h'][self.sequence_length-1:-1]
        
        # Dividir dados
        split_idx = int(0.7 * len(X))
        val_split = int(0.85 * len(X))
        
        X_train, X_val, X_test = X[:split_idx], X[split_idx:val_split], X[val_split:]
        y_train, y_val, y_test = y[:split_idx], y[split_idx:val_split], y[val_split:]
        
        # Treinar modelos
        models = self.train_ensemble_model(X_train, y_train, X_val, y_val)
        
        # Fazer predições
        predictions = {}
        for name, model in models.items():
            if name == 'random_forest':
                X_test_2d = X_test.reshape(X_test.shape[0], -1)
                predictions[name] = model.predict(X_test_2d)
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test).to(device)
                    outputs = model(X_tensor)
                    predictions[name] = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Combinar predições (voting)
        final_predictions = []
        for i in range(len(predictions['ensemble'])):
            votes = [predictions['ensemble'][i], 
                    predictions['transformer'][i], 
                    predictions['random_forest'][i]]
            final_predictions.append(max(set(votes), key=votes.count))
        
        # Backtesting
        test_df = combined_df.iloc[val_split + self.sequence_length:]
        
        backtester = AdvancedBacktester()
        probabilities = np.ones(len(final_predictions)) * 0.7  # Simulado
        
        results = backtester.run_backtest(test_df, final_predictions, probabilities)
        
        # Visualizar resultados
        self.plot_backtest_results(backtester, results)
        
        return results, models, backtester
    
    def create_sequences(self, data, target=None):
        """Cria sequências para modelos temporais"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            if target is not None:
                y.append(target[i + self.sequence_length])
        
        return np.array(X), np.array(y) if target is not None else None
    
    def plot_backtest_results(self, backtester, results):
        """Plota resultados do backtest"""
        
        if not backtester.equity_curve:
            return
        
        equity_df = pd.DataFrame(backtester.equity_curve)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity Curve
        axes[0, 0].plot(equity_df['timestamp'], equity_df['equity'])
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Portfolio Value')
        
        # Drawdown
        running_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - running_max) / running_max * 100
        axes[0, 1].fill_between(equity_df['timestamp'], drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown (%)')
        axes[0, 1].set_ylabel('Drawdown (%)')
        
        # Returns Distribution
        returns = equity_df['equity'].pct_change().dropna()
        axes[1, 0].hist(returns, bins=50, alpha=0.7)
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Returns')
        
        # Métricas
        metrics_text = f"""
        Total Return: {results['total_return']:.2%}
        Sharpe Ratio: {results['sharpe_ratio']:.2f}
        Max Drawdown: {results['max_drawdown']:.2%}
        Win Rate: {results['win_rate']:.2%}
        Profit Factor: {results['profit_factor']:.2f}
        Total Trades: {results['total_trades']}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def live_trading_monitor(self, symbol, models):
        """Monitor de trading em tempo real"""
        
        logger.info(f"Iniciando monitor de trading para {symbol}")
        
        while True:
            try:
                # Coletar dados atuais
                current_data = self.data_collector.get_multiple_timeframes(
                    symbol, ['1h'], days=3
                )
                
                if '1h' not in current_data:
                    time.sleep(60)
                    continue
                
                # Processar dados
                df = self.data_collector.add_advanced_indicators(current_data['1h'])
                
                # Preparar features
                feature_columns = [col for col in df.columns if col not in ['timestamp']]
                features = df[feature_columns].values[-self.sequence_length:]
                
                # Normalizar
                features_scaled = self.scalers['robust'].transform(features)
                
                # Fazer predição
                predictions = {}
                for name, model in models.items():
                    if name == 'random_forest':
                        X = features_scaled.reshape(1, -1)
                        predictions[name] = model.predict(X)[0]
                    else:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model.eval()
                        with torch.no_grad():
                            X = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)
                            output = model(X)
                            predictions[name] = torch.argmax(output, dim=1).item()
                
                # Combinir predições
                votes = list(predictions.values())
                final_prediction = max(set(votes), key=votes.count)
                
                # Log predição
                actions = ['HOLD', 'BUY', 'SELL']
                current_price = df['close'].iloc[-1]
                
                logger.info(f"{symbol} - Preço: {current_price:.2f}, "
                           f"Predição: {actions[final_prediction]}, "
                           f"Consenso: {votes.count(final_prediction)}/{len(votes)}")
                
                # Aguardar próxima verificação
                time.sleep(300)  # 5 minutos
                
            except Exception as e:
                logger.error(f"Erro no monitor: {e}")
                time.sleep(60)

# Função principal de exemplo
def main():
    """Função principal para executar o sistema completo"""
    
    # Configuração
    API_KEY = '55Zdzkxhw0qyZmxJnrMtqbhy6K9x2WiLw8AFRDpABOdwCntkFzSH7QBj0bnm5vJO'
    API_SECRET = '6utYlsoiB1a5U2SxtXaVnHkUQEjAtojnoqIwXKx7VvUkT50JzniYOgWrMzssFXDY'
    SYMBOL = 'BTCUSDT'
    
    # Inicializar sistema
    trading_system = AdvancedTradingSystem(API_KEY, API_SECRET)
    
    logger.info("Iniciando sistema de trading avançado...")
    
    try:
        # Executar backtest completo
        results, models, backtester = trading_system.run_comprehensive_backtest(
            None, SYMBOL
        )
        
        logger.info("Backtest completo!")
        logger.info(f"Retorno total: {results['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        # Salvar modelos
        for name, model in models.items():
            if name != 'random_forest':
                torch.save(model.state_dict(), f'{SYMBOL}_{name}_model.pth')
            else:
                with open(f'{SYMBOL}_{name}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
        
        logger.info("Modelos salvos!")
        
        # Iniciar monitor em tempo real (comentado para exemplo)
        # trading_system.live_trading_monitor(SYMBOL, models)
        
    except Exception as e:
        logger.error(f"Erro na execução: {e}")

if __name__ == "__main__":
    main()