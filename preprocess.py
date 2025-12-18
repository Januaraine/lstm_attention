import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 计算滚动波动率，为计算因子铺垫
def rolling_volatility(series, window, direction):
    
    def vol_func(x):
        if direction == 'down':
            # 筛选窗口内的负收益
            neg_returns = x[x < 0]
            if len(neg_returns) < 2:  # 至少需要2个点计算标准差
                return np.nan
            return neg_returns.std()
        elif direction == 'up':
            # 筛选窗口内的正收益
            pos_returns = x[x > 0]
            if len(pos_returns) < 2:  # 至少需要2个点计算标准差
                return np.nan
            return pos_returns.std()
        else:
            return np.nan
    
    # 应用滚动计算
    return series.rolling(window=window, min_periods=1).apply(vol_func, raw=False)


def add_factors(file):
    # 计算未来收益因子
    file['return'] = file.groupby('s_info_windcode')['s_dq_adjclose'].pct_change()
    file['future_return'] = file.groupby('s_info_windcode')['s_dq_adjclose'].pct_change().shift(-20)

    # 1个月收益率
    file['mmt_normal_M'] = (
        file['s_dq_adjclose'] /
        file.groupby('s_info_windcode')['s_dq_adjclose'].shift(20) - 1
    )

    # 1年收益率
    file['mmt_normal_A'] = (
        file['s_dq_adjclose'] /
        file.groupby('s_info_windcode')['s_dq_adjclose'].shift(240) - 1
    )

    # 相对均价的1个月收益率
    file['avg20'] = file.groupby('s_info_windcode')['s_dq_adjclose'].transform(lambda x: x.rolling(20).mean())
    file['mmt_avg_M'] = file['s_dq_adjclose'] / file['avg20']

    # 相对均价的1年收益率
    file['avg240'] = file.groupby('s_info_windcode')['s_dq_adjclose'].transform(lambda x: x.rolling(240).mean())
    file['mmt_avg_A'] = file['s_dq_adjclose'] / file['avg240']


    # 1个月日内动量
    file['intraday_return'] = (file['s_dq_adjclose'] - file['s_dq_adjopen']) / file['s_dq_adjopen']
    file['mmt_intraday_M'] = file.groupby('s_info_windcode')['intraday_return'].transform(lambda x: x.rolling(20).sum())
    # 1年日内动量
    file['mmt_intraday_A'] = file.groupby('s_info_windcode')['intraday_return'].transform(lambda x: x.rolling(240).sum())


    # 1个月隔夜动量
    prev_close = file.groupby('s_info_windcode')['s_dq_adjclose'].shift(1)
    file['overnight_return'] = (file['s_dq_adjopen'] - prev_close) / prev_close
    file['mmt_overnight_M'] = file.groupby('s_info_windcode')['overnight_return'].transform(lambda x: x.rolling(20).sum())
    # 1年隔夜动量
    file['mmt_overnight_A'] = file.groupby('s_info_windcode')['overnight_return'].transform(lambda x: x.rolling(240).sum())


    
    file['daily_return'] = file.groupby('s_info_windcode')['s_dq_adjclose'].pct_change()
    
    # 1个月路径调整动量
    file['cum_return_20'] = file.groupby('s_info_windcode')['daily_return'] \
        .transform(lambda x: (1 + x).rolling(20).apply(np.prod, raw=True) - 1)

    file['abs_return_sum_20'] = file.groupby('s_info_windcode')['daily_return'] \
        .transform(lambda x: x.abs().rolling(20).sum())

    file['mmt_route_M'] = file['cum_return_20'] / file['abs_return_sum_20']
    
    # 1年路径调整动量
    file['cum_return_240'] = file.groupby('s_info_windcode')['daily_return'] \
        .transform(lambda x: (1 + x).rolling(240).apply(np.prod, raw=True) - 1)

    file['abs_return_sum_240'] = file.groupby('s_info_windcode')['daily_return'] \
        .transform(lambda x: x.abs().rolling(240).sum())

    file['mmt_route_A'] = file['cum_return_240'] / file['abs_return_sum_240']


    # 1个月信息离散度动量
    file['up_day'] = (file['daily_return'] > 0).astype(int)
    file['down_day'] = (file['daily_return'] < 0).astype(int)

    file['up_ratio_20'] = file.groupby('s_info_windcode')['up_day'].transform(lambda x: x.rolling(20).mean())
    file['down_ratio_20'] = file.groupby('s_info_windcode')['down_day'].transform(lambda x: x.rolling(20).mean())
    file['mmt_discrete_M'] = file['up_ratio_20'] - file['down_ratio_20']

    # 1年信息离散度动量
    file['up_ratio_240'] = file.groupby('s_info_windcode')['up_day'].transform(lambda x: x.rolling(240).mean())
    file['down_ratio_240'] = file.groupby('s_info_windcode')['down_day'].transform(lambda x: x.rolling(240).mean())
    file['mmt_discrete_A'] = file['up_ratio_240'] - file['down_ratio_240']


    # 1个月横截面rank动量
    file['daily_rank'] = file.groupby('trade_dt')['daily_return'].rank(pct=True)
    file['mmt_sec_rank_M'] = file.groupby('s_info_windcode')['daily_rank'].transform(lambda x: x.rolling(20).mean())
    # 1年横截面rank动量
    file['mmt_sec_rank_A'] = file.groupby('s_info_windcode')['daily_rank'].transform(lambda x: x.rolling(240).mean())


    # 1个月时序rank动量
    file['price_rank'] = file.groupby('s_info_windcode')['s_dq_adjclose'] \
        .transform(lambda x: x.rolling(240).rank(pct=True))
    file['mmt_time_rank_M'] = file.groupby('s_info_windcode')['price_rank'].transform(lambda x: x.rolling(20).mean())


    # x个月波动率
    file['vol_std_1M'] = file.groupby('s_info_windcode')['return'].transform(lambda x: x.rolling(20).std())
    file['vol_std_3M'] = file.groupby('s_info_windcode')['return'].transform(lambda x: x.rolling(60).std())
    file['vol_std_6M'] = file.groupby('s_info_windcode')['return'].transform(lambda x: x.rolling(120).std())

    # x个月上行波动率 + x个月下行波动率
    for w in [20, 60, 120]:
        file[f'vol_up_std_{w}'] = file.groupby('s_info_windcode')['return'].transform(lambda x: rolling_volatility(x, w, 'up'))
        file[f'vol_down_std_{w}'] = file.groupby('s_info_windcode')['return'].transform(lambda x: rolling_volatility(x, w, 'down'))

    # x个月日内振幅 + x个月日内振幅标准差
    file['high_low_ratio'] = file['s_dq_adjhigh'] / file['s_dq_adjlow']

    for w in [20, 60, 120]:
        file[f'vol_highlow_avg_{w}'] = file.groupby('s_info_windcode')['high_low_ratio'].transform(lambda x: x.rolling(w).mean())
        file[f'vol_highlow_std_{w}'] = file.groupby('s_info_windcode')['high_low_ratio'].transform(lambda x: x.rolling(w).std())


    # x个月上影线均值 + x个月上影线标准差
    file['upshadow'] = (file['s_dq_adjhigh'] - file[['s_dq_adjopen', 's_dq_adjclose']].max(axis=1)) / file['s_dq_adjhigh']
    file['downshadow'] = (file[['s_dq_adjopen', 's_dq_adjclose']].min(axis=1) - file['s_dq_adjlow']) / file['s_dq_adjlow']

    for name in ['upshadow', 'downshadow']:
        for w in [20, 60, 120]:
            file[f'vol_{name}_avg_{w}'] = file.groupby('s_info_windcode')[name].transform(lambda x: x.rolling(w).mean())
            file[f'vol_{name}_std_{w}'] = file.groupby('s_info_windcode')[name].transform(lambda x: x.rolling(w).std())


    # x个月威廉上影线均值 + x个月威廉上影线标准差
    file['w_upshadow'] = (file['s_dq_adjhigh'] - file['s_dq_adjclose']) / file['s_dq_adjhigh']
    for w in [20, 60, 120]:
        file[f'vol_w_upshadow_avg_{w}'] = file.groupby('s_info_windcode')['w_upshadow'].transform(lambda x: x.rolling(w).mean())
        file[f'vol_w_upshadow_std_{w}'] = file.groupby('s_info_windcode')['w_upshadow'].transform(lambda x: x.rolling(w).std())

    
    # volume=0 → fill
    file['volume_filled'] = file.groupby('s_info_windcode')['s_dq_volume'].transform(lambda x: x.replace(0, np.nan).ffill().bfill())
    
    # x个月成交波动比
    for w in [20, 60, 120]:
        # 成交量均值
        vol_avg = file.groupby('s_info_windcode')['s_dq_volume'].transform(lambda x: x.rolling(w).mean())
        # 收益率标准差
        ret_std = file.groupby('s_info_windcode')['return'].transform(lambda x: x.rolling(w).std())
        file[f'liq_vstd_{w}'] = (vol_avg / ret_std).replace([np.inf, -np.inf], np.nan)


    # x个月Amihud非流动因子 + 标准差
    file['return_filled'] = file.groupby('s_info_windcode')['return'].transform(lambda x: x.ffill().bfill())
    file['amihud'] = (file['return_filled'].abs() / file['volume_filled']).replace([np.inf, -np.inf], np.nan)

    for w in [20, 60, 120]:
        file[f'liq_amihud_avg_{w}'] = file.groupby('s_info_windcode')['amihud'].transform(lambda x: x.rolling(w).mean())
        file[f'liq_amihud_std_{w}'] = file.groupby('s_info_windcode')['amihud'].transform(lambda x: x.rolling(w).std())


    # x个月最短路径非流动因子 + 标准差
    file['shortcut_path'] = 2 * (file['s_dq_adjhigh'] - file['s_dq_adjlow']) - abs(file['s_dq_adjopen'] - file['s_dq_adjclose'])

    for w in [20,60,120]:
        file['liq_shortcut_avg_{w}'] = (file['shortcut_path'] / file['s_dq_volume']).transform(lambda x: x.rolling(w).mean())
        file['liq_shortcut_std_{w}'] = (file['shortcut_path'] / file['s_dq_volume']).transform(lambda x: x.rolling(w).std())


    return file




# 技术指标
def compute_tech_factors(file):
    file = file.copy()

    # 日收益率（基于 preclose）
    file["ret"] = file["s_dq_preclose"].pct_change()        # 相对变化率

    # high-low relative range
    file["hl_range"] = (file["s_dq_high"] - file["s_dq_low"]) / file["s_dq_preclose"]

    # 成交量变化率
    file["vol_chg"] = file["s_dq_volume"].pct_change()

    # 隐含成交价格（注意避免 /0）
    file["avg_price"] = file["s_dq_amount"] / file["s_dq_volume"].replace(0, np.nan)

    # Rolling volatility
    file["volatility"] = file["ret"].rolling(10).std()

    # RSI 预防除 0
    delta = file["s_dq_preclose"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    up_mean = up.rolling(14).mean()
    down_mean = down.rolling(14).mean().replace(0, 1e-6)
    rs = up_mean / down_mean
    file["rsi"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = file["s_dq_preclose"].ewm(span=12).mean()
    ema26 = file["s_dq_preclose"].ewm(span=26).mean()
    file["macd"] = ema12 - ema26


    file = add_factors(file)

    # 清理 inf
    file = file.replace([np.inf, -np.inf], np.nan)

    return file





def load_and_preprocess(path, windcode, window_size):
    file = pd.read_csv(path)
    
    file = file[file['s_info_windcode'] == windcode].copy()
    if file.empty:
        raise ValueError(f"No data found for ticker {windcode} in {path}")

    file['trade_dt'] = pd.to_datetime(file['trade_dt'],format='%Y%m%d')
    file.sort_values('trade_dt', inplace=True)

    # 计算技术因子
    file = compute_tech_factors(file)

    # rolling 会产生NaN，这里再填补
    file = file.bfill().ffill()

    features = [
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

    # 转numpy
    data = file[features].values.astype(np.float32)
    price = file['s_dq_preclose'].values.astype(np.float32)

    # 标签：下一天 preclose
    next_price = np.roll(price, -1)    # 每个元素现在对应的是下一天的价格
    next_ret = (next_price - price) / price


    data = data[:-1]        # 删除最后一天无标签数据
    next_ret = next_ret[:-1]

    # 划分 dataset
    n = len(data)
    train_size = int(n*0.7)
    val_size = int(n*0.15)

    train_data = data[:train_size]
    train_next_ret = next_ret[:train_size]

    val_data = data[train_size:train_size+val_size]
    val_next_ret = next_ret[train_size:train_size+val_size]

    test_data = data[train_size+val_size:]
    test_next_ret = next_ret[train_size+val_size:]

    # Normalization
    # ❌ data = scaler.fit_transform(data)  未来信息泄漏
    feature_scaler = StandardScaler()
    train_data = feature_scaler.fit_transform(train_data)   # 计算均值和标准差只基于训练集
    val_data = feature_scaler.transform(val_data)
    test_data = feature_scaler.transform(test_data)

    label_scaler = StandardScaler()
    train_next_ret = label_scaler.fit_transform(train_next_ret.reshape(-1,1)).reshape(-1)    
    val_next_ret = label_scaler.transform(val_next_ret.reshape(-1,1)).reshape(-1)
    test_next_ret = label_scaler.transform(test_next_ret.reshape(-1,1)).reshape(-1)

    # slicing，天数
    def create_windows(data, next_ret, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(next_ret[i+window_size]) 
        return np.array(X), np.array(y)

    X_train, y_train = create_windows(train_data, train_next_ret, window_size)
    X_val, y_val = create_windows(val_data, val_next_ret, window_size)
    X_test, y_test = create_windows(test_data, test_next_ret, window_size)

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            feature_scaler, label_scaler, len(features))
