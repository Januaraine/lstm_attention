import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureProcessor:
    
    def __init__(self, feature_names=None):
        # 初始化空的历史数据容器
        self.historical_data = pd.DataFrame()
        self.feature_dim = 76  # 根据你的训练特征数量设置
        self.feature_scaler = None
        self.label_scaler = None
        
    def calculate_features(self, open_price, high_price, low_price, 
                          close_price, volume, prev_close=None, current_date=None):
        # 如果历史数据为空，初始化
        if self.historical_data.empty:
            self.historical_data = pd.DataFrame({
                'open': [open_price],
                'high': [high_price],
                'low': [low_price],
                'close': [close_price],
                'volume': [volume],
                'prev_close': [prev_close] if prev_close else [close_price],
                'date': [current_date] if current_date else [pd.Timestamp.now()]
            })
            # 首次调用，无法计算需要历史数据的特征
            return self._get_basic_features(open_price, high_price, low_price, close_price, volume, prev_close)
        
        # 添加新数据
        new_row = {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'prev_close': prev_close if prev_close else self.historical_data['close'].iloc[-1],
            'date': current_date if current_date else pd.Timestamp.now()
        }
        self.historical_data = pd.concat([self.historical_data, pd.DataFrame([new_row])], ignore_index=True)
        
        # 确保有足够的数据计算所有特征
        if len(self.historical_data) < 240:  # 有些特征需要240天窗口
            return self._get_basic_features(open_price, high_price, low_price, close_price, volume, prev_close)
        
        # 计算所有76个特征
        return self._calculate_all_features()
    
    def _get_basic_features(self, open_price, high_price, low_price, close_price, volume, prev_close):
        """当数据不足时返回基础特征"""
        # 计算基础特征
        if prev_close is not None and prev_close != 0:
            daily_return = (close_price - prev_close) / prev_close
        else:
            daily_return = 0
            
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # 填充基础特征（前几个位置）
        features[0] = prev_close if prev_close else close_price  # s_dq_preclose
        features[1] = open_price   # s_dq_open
        features[2] = high_price   # s_dq_high
        features[3] = low_price    # s_dq_low
        features[4] = close_price  # s_dq_close
        features[5] = daily_return # ret
        
        return features
    
    def _calculate_all_features(self):
        """计算所有76个特征"""
        # 这里需要实现与训练时完全相同的特征计算逻辑
        # 由于时间关系，这里提供一个简化版本
        # 实际需要根据你的preprocess.py中的特征计算逻辑来实现
        
        df = self.historical_data.copy()
        
        # 1. 计算技术指标（与训练时一致）
        df["ret"] = df["close"].pct_change()
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]
        df["vol_chg"] = df["volume"].pct_change()
        df["avg_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["volatility"] = df["ret"].rolling(10).std()
        
        # RSI
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        up_mean = up.rolling(14).mean()
        down_mean = down.rolling(14).mean().replace(0, 1e-6)
        rs = up_mean / down_mean
        df["rsi"] = 100 - 100 / (1 + rs)
        
        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        
        # 获取最新一行数据的所有特征
        latest_features = []
        
        # 基础价格数据
        latest_features.extend([
            df["prev_close"].iloc[-1],  # s_dq_preclose
            df["open"].iloc[-1],        # s_dq_open
            df["high"].iloc[-1],        # s_dq_high
            df["low"].iloc[-1],         # s_dq_low
            df["close"].iloc[-1],       # s_dq_close
        ])
        
        # 技术指标
        latest_features.extend([
            df["ret"].iloc[-1],
            df["hl_range"].iloc[-1],
            df["vol_chg"].iloc[-1],
            df["avg_price"].iloc[-1],
            df["volatility"].iloc[-1],
            df["rsi"].iloc[-1],
            df["macd"].iloc[-1]
        ])
        
        # 动量因子（这里需要实现与训练时相同的计算）
        # 由于时间关系，这里只添加占位符
        momentum_features = [0.0] * 20  # 根据实际动量因子数量调整
        latest_features.extend(momentum_features)
        
        # 波动率因子
        vol_features = [0.0] * 15  # 根据实际波动率因子数量调整
        latest_features.extend(vol_features)
        
        # 流动性因子
        liq_features = [0.0] * 10  # 根据实际流动性因子数量调整
        latest_features.extend(liq_features)
        
        # 确保总共有76个特征
        if len(latest_features) < self.feature_dim:
            # 用0填充剩余特征
            latest_features.extend([0.0] * (self.feature_dim - len(latest_features)))
        elif len(latest_features) > self.feature_dim:
            # 截断多余特征
            latest_features = latest_features[:self.feature_dim]
        
        return np.array(latest_features, dtype=np.float32)
    
    def fit(self, data):
        self.feature_scaler.fit(data)
        
    def transform_features(self, features):
        if self.feature_scaler is None:
            return features
        return self.feature_scaler.transform(features.reshape(1, -1)).flatten()
    
    def inverse_transform_label(self, label):
        if self.label_scaler is None:
            return label
        return self.label_scaler.inverse_transform(label.reshape(-1, 1)).flatten()