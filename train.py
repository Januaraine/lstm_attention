import numpy as np
from scipy.stats import spearmanr # 推荐使用 scipy 的 spearmanr，因为它能处理 NaN
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from LSTM_attention_Model import LSTM_attention_Model
from preprocess import load_and_preprocess
import pickle # 导入 pickle 以保存 scaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

FILE_PATH = "xxxx.csv"
WINDCODE = 'xxx.SZ'

# Data preprocess
(X_train, y_train, X_val, y_val, _, _,
 feature_scaler, label_scaler, feature_dim) = load_and_preprocess(
    FILE_PATH, WINDCODE, window_size=20)

# 转换成tensor 
# n维数组，也称为张量（tensor），支持 GPU 加速和自动梯度计算
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)


batch_size = 64

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size, shuffle=False)
# shuffle: 打乱数据顺序


# Training
model = LSTM_attention_Model(feature=feature_dim).to(device) 

# class CombinedLoss(nn.Module):
#     def __init__(self, alpha=0.7):
#         super().__init__()
#         self.alpha = alpha
#         self.mse = nn.MSELoss()
    
#     def direction_loss(self, pred, target):
#         # 鼓励预测和真实值同号
#         # 使用交叉熵风格的损失，关注方向
#         return torch.mean(torch.log(1 + torch.exp(-pred * target)))
    
#     def forward(self, pred, target):
#         mse = self.mse(pred, target)
#         dir_loss = self.direction_loss(pred, target)
#         return self.alpha * mse + (1 - self.alpha) * dir_loss


# criterion = nn.MarginRankingLoss(margin=0.2) # 增加 margin 有助于学习差异
# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()  # Hube 对异常值不敏感
# criterion = CombinedLoss(alpha=0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)    # 权重和偏置


epochs = 100

for epoch in range(epochs):
    model.train()

    epoch_loss = 0
    correct_dir = 0
    total_dir = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch).squeeze()

        loss = criterion(preds, y_batch)
        loss.backward()     # 给每个 parameter 算梯度
        optimizer.step()    # 梯度更新
        # Attention 不会死盯极端时间点
        # Feature Gate 不会因为异常值而失控
        # LSTM 更倾向学稳定模式

        epoch_loss += loss.item()

        # ---- train direction ----
        p = preds.detach().cpu().numpy()
        y = y_batch.detach().cpu().numpy()

        pred_dir = np.sign(p[1:] - p[:-1])
        real_dir = np.sign(y[1:] - y[:-1])

        correct_dir += np.sum(pred_dir == real_dir)
        total_dir += len(real_dir)


    epoch_dir_acc = correct_dir / total_dir

    print(f"Epoch {epoch+1}/{epochs}  "
          f"Loss: {epoch_loss/len(train_loader):.6f}  "
          f"DirAcc: {epoch_dir_acc:.4f}")



# Validation
model.eval()
val_loss = 0
val_dir_correct = 0
val_dir_total = 0

with torch.no_grad():       # 只是看模型效果
    for Xb, yb in val_loader:
        pr = model(Xb).squeeze()
        val_loss += criterion(pr, yb).item()

        pr_cpu = pr.cpu().numpy()
        yb_cpu = yb.cpu().numpy()

        pred_dir = np.sign(pr_cpu[1:] - pr_cpu[:-1])
        real_dir = np.sign(yb_cpu[1:] - yb_cpu[:-1])
        val_dir_correct += np.sum(pred_dir == real_dir)
        val_dir_total += len(real_dir)

val_dir_acc = val_dir_correct / val_dir_total

print(f"Epoch {epoch+1}/{epochs}  "
        f"TrainLoss: {epoch_loss/len(train_loader):.6f}  "
        f"TrainDir: {epoch_dir_acc:.4f}  "
        f"ValLoss: {val_loss/len(val_loader):.6f}  "
        f"ValDir: {val_dir_acc:.4f}")





# 保存
model_path = "lstm_etf.pth"

torch.save({
    'model_state_dict': model.state_dict(),
    'feature_dim': feature_dim  # 保存特征维度
}, model_path)
print("Model saved!")

# 保存 scaler（pickle）
import pickle
pickle.dump(feature_scaler, open("feature_scaler.pkl", "wb"))
pickle.dump(label_scaler, open("label_scaler.pkl", "wb"))
print("Scaler saved!")