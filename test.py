import torch
from LSTM_attention_Model import LSTM_attention_Model
from preprocess import load_and_preprocess
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle # Need pickle to load scalers if they are separated

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

def eval_on_loader(model, loader, device, label_scaler):
    """Evaluates the model on the full sequence from the loader."""
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            # out: [batch_size, 1] or [batch_size]
            out = model(Xb).squeeze().cpu().numpy() 
            preds.append(out)
            trues.append(yb.numpy())
    
    preds = np.concatenate(preds, axis=0) # Normalized predictions
    trues = np.concatenate(trues, axis=0) # Normalized true returns
    
    # 转换为实际收益率
    preds_real = label_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
    trues_real = label_scaler.inverse_transform(trues.reshape(-1, 1)).reshape(-1)
    
    return preds_real, trues_real



FILE_PATH = "xxxx.csv"
WINDCODE = 'xxx.SZ'

# Data preprocess -> test only
(_, _, _, _, X_test, y_test,
 feature_scaler, label_scaler, feature_dim) = load_and_preprocess(
    FILE_PATH, WINDCODE, window_size=20)

# tensors for DataLoader
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)


# OOM FIX: 使用 DataLoader 进行分批推理 
batch_size = 64 
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

# Model load: lstm_hidden
model = LSTM_attention_Model(feature=feature_dim).to(device)
# 加载在 train.py 中保存的最佳模型
checkpoint = torch.load("lstm_etf.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Get real predictions and real true returns
preds_real, y_test_real = eval_on_loader(model, test_loader, device, label_scaler)

# 评估
rmse = np.sqrt(np.mean((preds_real - y_test_real)**2))

# Direction Accuracy Timely (相邻时间点的变化方向，往上拐还是往下拐)
direction_acc_time = np.mean(
    np.sign(preds_real[1:] - preds_real[:-1]) ==
    np.sign(y_test_real[1:] - y_test_real[:-1])
)

# Direction Accuracy (方向正负, next_ret > 0)
# 排除真实收益率为 0 的情况
non_zero_true = y_test_real != 0
direction_acc = np.mean(np.sign(preds_real[non_zero_true]) == np.sign(y_test_real[non_zero_true]))

# 夏普比率 (假设无风险利率为0)
signal = np.sign(preds_real)
strategy_ret = signal * y_test_real    # y_real 是 next-day return

sharpe = np.mean(strategy_ret) / np.std(strategy_ret + 1e-8)

print("RMSE:", rmse)
print("Direction Accuracy Timely:", direction_acc_time)
print("Direction Accuracy:", direction_acc)
print("Sharpe Ratio:", sharpe)