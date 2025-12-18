import backtrader as bt
import numpy as np
import torch
import pickle
import pandas as pd
from LSTM_attention_Model import LSTM_attention_Model
from preprocess import compute_tech_factors 


FEATURES_NAME = [
        's_dq_preclose', 's_dq_open', 's_dq_high', 's_dq_low', 's_dq_close',
        's_dq_adjpreclose', 's_dq_adjopen', 's_dq_adjhigh', 's_dq_adjlow', 's_dq_adjclose',
        # 技术指标
        'ret','hl_range','vol_chg','avg_price','volatility','rsi','macd',
        # more features
        # 动量因子
        'return', 'future_return', 
        'mmt_normal_M', 'mmt_normal_A', 'mmt_avg_M', 'mmt_avg_A',
        'mmt_intraday_M', 'mmt_intraday_A', 'mmt_overnight_M', 'mmt_overnight_A',
        'mmt_route_M', 'mmt_route_A', 'mmt_discrete_M', 'mmt_discrete_A',
        'mmt_sec_rank_M', 'mmt_sec_rank_A', 'mmt_time_rank_M',

        # 波动率因子
        'vol_std_1M', 'vol_std_3M', 'vol_std_6M',
        'vol_up_std_20', 'vol_up_std_60', 'vol_up_std_120',
        'vol_down_std_20', 'vol_down_std_60', 'vol_down_std_120',
        
        # 日内振幅
        'vol_highlow_avg_20', 'vol_highlow_avg_60', 'vol_highlow_avg_120',
        'vol_highlow_std_20', 'vol_highlow_std_60', 'vol_highlow_std_120',
        
        # 影线因子
        'vol_upshadow_avg_20', 'vol_upshadow_avg_60', 'vol_upshadow_avg_120',
        'vol_upshadow_std_20', 'vol_upshadow_std_60', 'vol_upshadow_std_120',
        'vol_downshadow_avg_20', 'vol_downshadow_avg_60', 'vol_downshadow_avg_120',
        'vol_downshadow_std_20', 'vol_downshadow_std_60', 'vol_downshadow_std_120',
        'vol_w_upshadow_avg_20', 'vol_w_upshadow_avg_60', 'vol_w_upshadow_avg_120',
        'vol_w_upshadow_std_20', 'vol_w_upshadow_std_60', 'vol_w_upshadow_std_120',
        
        # 流动性因子
        'liq_vstd_20', 'liq_vstd_60', 'liq_vstd_120',
        'liq_amihud_avg_20', 'liq_amihud_avg_60', 'liq_amihud_avg_120',
        'liq_amihud_std_20', 'liq_amihud_std_60', 'liq_amihud_std_120'

    ]



class Strategy(bt.Strategy):
    params = (
        ('window_size', 20),                                # 与训练时相同的窗口大小
        ('position_size', 0.1),                             # 每次交易仓位比例 (10%)
        ('stop_loss', 0.05),                                # 止损比例
        ('take_profit', 0.08),                              # 止盈比例 
        ('signal_threshold', 0.3), 
        ('model_path', 'lstm_etf.pth'),
        ('feature_scaler_path', 'feature_scaler.pkl'),
        ('label_scaler_path', 'label_scaler.pkl'),
    )
    
    def __init__(self):
        # 初始化数据窗口
        self.data_window = []
        self.order = None
        self.position_opened = False
        
        # 加载模型和scaler
        self.load_model_and_scalers()
        
        # 技术指标
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=10, plot=False)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=30, plot=False)
        self.rsi = bt.indicators.RSI(self.data.close, period=14, plot=False)
        
        # 记录
        self.trade_count = 0
        self.win_count = 0
        self.total_return = 0
        
        # 确保 FEATURES_NAME 在 data 中存在（记录不存在的列）
        self.available_feature_names = []
        self.missing_feature_names = []
        for name in FEATURES_NAME:
            # Backtrader 会把 pandas 的列变成 data.<colname>
            if hasattr(self.data, name):
                self.available_feature_names.append(name)
            else:
                self.missing_feature_names.append(name)

        if self.missing_feature_names:
            print("警告：以下特征列的数据缺失，将填充为 NaN:",
                  self.missing_feature_names)

        # 直接存 name 列表，读取时用 getattr(self.data, name)[0]
        self.features_names_used = self.available_feature_names + self.missing_feature_names


        self.prev_close = None  # 记录前一天的收盘价
        
    def load_model_and_scalers(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载scaler
        with open(self.params.feature_scaler_path, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(self.params.label_scaler_path, 'rb') as f:
            self.label_scaler = pickle.load(f)
        
        # 加载模型
        # 加载模型检查点（包含模型状态和特征维度）
        checkpoint = torch.load(self.params.model_path, map_location='cpu')
        
        # 从检查点获取特征维度
        feature_dim = checkpoint['feature_dim']
        self.model = LSTM_attention_Model(feature_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)
        print("加载成功！")
            
 
    def next(self):
        # 如果有未完成的订单，先等待    
        if self.order:
            return
        
        # 收集数据到窗口
        self.collect_data()
        
        # 如果窗口大小足够，进行预测
        if len(self.data_window) >= self.params.window_size:
            # 生成交易信号
            signal = self.generate_signal()
            
            # 检查止损止盈
            self.check_stop_loss_take_profit()
            
            # 执行交易
            self.execute_trade(signal)
        

    def collect_data(self):

        features_list = []
        for name in self.features_names_used:
            # 判断 Backtrader 的 data 对象里，是否存在名为 name 的数据线（line）
            if hasattr(self.data, name):
                val = getattr(self.data, name)[0]  # 当前 bar 的值
            else:
                # 列不存在 -> 填 NaN（后面可用 np.nan_to_num 或 scaler 处理）
                val = np.nan
            features_list.append(val)
    
        features = np.array(features_list, dtype=np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 更新前一天的收盘价 (虽然不再用于特征计算，但仍用于止损止盈或 log)
        # 使用 Backtrader 的 close 线
        self.prev_close = self.data.close[0]
        
        # 添加到数据窗口
        self.data_window.append(features)
        
        # 保持窗口大小
        if len(self.data_window) > self.params.window_size:
            self.data_window.pop(0)
        

    # 使用模型生成交易信号
    def generate_signal(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 准备数据
        window_data = np.array(self.data_window, dtype=np.float32)
        
        # 特征标准化
        window_scaled = self.feature_scaler.transform(window_data)
        
        # 转换为模型输入格式 [batch, seq_len, features]
        input_data = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0)
        input_data = input_data.to(device)

        # 模型预测（预测下一天的涨跌额）
        with torch.no_grad():
            pred_tensor = self.model(input_data)

        preds = pred_tensor.squeeze().detach().cpu().item()
        
        # 反标准化得到真实的下一天涨跌额预测
        preds_real = self.label_scaler.inverse_transform(
            np.array([[preds]]).reshape(-1, 1)
        )[0][0]
        
        # 计算预测的下一天涨跌幅
        current_close = self.data.close[0]
        predicted_change_pct = preds_real / current_close * 100 if current_close != 0 else 0
        
        # 生成信号 - 基于预测的下一天表现
        signal = 0
        if predicted_change_pct > self.params.signal_threshold:  # 预测明天上涨超过
            signal = 1  # 买入信号
        elif predicted_change_pct < -self.params.signal_threshold:  # 预测明天下跌超过
            signal = -1  # 卖出信号
        
        # 记录预测值，可用于分析
        if not hasattr(self, 'predictions'):
            self.preds = []
        self.preds.append(predicted_change_pct)
        
        return signal

    
    # 执行交易
    def execute_trade(self, signal):
       
        current_price = self.data.close[0]
        
        # 卖出信号
        if signal == -1 and self.position.size > 0:
            self.close()
            self.position_opened = False
            self.log(f"SELL, Price: {current_price:.2f}")
        
        # 买入信号
        elif signal == 1 and self.position.size == 0:
            # 计算购买数量
            cash = self.broker.get_cash()
            position_value = cash * self.params.position_size
            size = int(position_value / current_price)
            
            if size > 0:
                self.buy(size=size)
                self.position_opened = True
                self.entry_price = current_price
                self.log(f"BUY, Price: {current_price:.2f}, Size: {size}")
                self.trade_count += 1
    
    
    # 检查止损止盈
    def check_stop_loss_take_profit(self):
        if not self.position_opened or self.position.size == 0:
            return
        
        current_price = self.data.close[0]
        profit_pct = (current_price - self.entry_price) / self.entry_price
        
        # 止损
        if profit_pct <= -self.params.stop_loss:
            self.close()
            self.position_opened = False
            self.log(f"STOP LOSS, Price: {current_price:.2f}, Loss: {profit_pct:.2%}")
        
        # 止盈
        elif profit_pct >= self.params.take_profit:
            self.close()
            self.position_opened = False
            self.log(f"TAKE PROFIT, Price: {current_price:.2f}, Profit: {profit_pct:.2%}")
            self.win_count += 1
    
    # 交易通知
    def notify_trade(self, trade):
        if trade.isclosed:
            self.total_return += trade.pnl
            self.log(f'交易盈亏, 毛利润 {trade.pnl:.2f}, 净利润 {trade.pnlcomm:.2f}')
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    # 回测结束
    def stop(self):
        win_rate = self.win_count / max(self.trade_count, 1) * 100
        self.log(f'回测结束')
        self.log(f'总交易次数: {self.trade_count}')
        self.log(f'胜率: {win_rate:.2f}%')
        self.log(f'总收益: {self.broker.get_value() - 100000:.2f}')
        self.log(f'年化收益: {self.get_annual_return():.2f}%')
        print(np.percentile(self.preds, [1, 5, 50, 95, 99]))

    
    def get_annual_return(self):
        """计算年化收益率"""
        years = len(self) / 252  # 假设一年252个交易日
        if years > 0:
            total_return = (self.broker.get_value() / 100000) - 1
        #     annual_return = (1 + total_return) ** (1 / years) - 1
        #     return annual_return * 100
        # return 0
            base = 1 + total_return
            if base <= 0:
                # 资金亏光，无法计算年化收益率，直接返回总亏损
                return total_return * 100 
                
            annual_return = base ** (1 / years) - 1
            return annual_return * 100
        return 0


# 设置回测环境
cerebro = bt.Cerebro()

# 设置初始资金
cerebro.broker.setcash(100000.0)

# 设置佣金
cerebro.broker.setcommission(commission=0.001)  # 0.1%佣金

# 加载策略
cerebro.addstrategy(Strategy)

# 添加分析器
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')


# =========================================================================================

# 加载数据
FILE_PATH = "xxxx.csv"
WINDCODE = 'xxx.SZ'

file = pd.read_csv(FILE_PATH)
    
file = file[file['s_info_windcode'] == WINDCODE].copy()
if file.empty:
    raise ValueError(f"No data found for ticker {WINDCODE} in {FILE_PATH}")

file['trade_dt'] = pd.to_datetime(file['trade_dt'],format='%Y%m%d')
file.sort_values('trade_dt', inplace=True)


# class PandasData_Extend(bt.feeds.PandasData):
#     lines = tuple(FEATURES_NAME)
#     params = tuple([(name, -1) for name in FEATURES_NAME])

# 加载与计算
file = pd.read_csv(FILE_PATH)
file = file[file['s_info_windcode'] == WINDCODE].copy()
file['trade_dt'] = pd.to_datetime(file['trade_dt'], format='%Y%m%d')
file.sort_values('trade_dt', inplace=True)


df = compute_tech_factors(file) 
df = df.ffill().bfill().fillna(0.0) # 彻底杜绝 NaN

# 列名适配
df = df.rename(columns={
    's_dq_open': 'open',
    's_dq_close': 'close',
    's_dq_high': 'high',
    's_dq_low': 'low',
    's_dq_volume': 'volume',
    's_dq_amount': 'amount',
    's_dq_preclose': 'preclose'
})

# 喂入 Cerebro

data_feed = bt.feeds.PandasData( 
    dataname=df, 
    datetime='trade_dt', # 使用trade_dt作为日期列 
    open='open', 
    high='high', 
    low='low', 
    close='close', 
    volume='volume', 
    # openinterest=None 
)

# data_feed = PandasData_Extend(
#     dataname=df,
#     datetime='trade_dt',
#     open='open', 
#     high='high', 
#     low='low', 
#     close='close', 
#     volume='volume'
# )

cerebro.adddata(data_feed)

print('初始资金: %.2f' % cerebro.broker.getvalue())

# 运行回测
results = cerebro.run()

print('最终资金: %.2f' % cerebro.broker.getvalue())

# 绘制图表
# cerebro.plot(style='candlestick')
cerebro.plot(
    style='candlestick',
    volume=False,      # 关掉成交量
    barup='red',
    bardown='green'
)

