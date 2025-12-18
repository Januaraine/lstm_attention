from torch import nn

class LSTM_attention_Model(nn.Module):
    def __init__(self, feature, lstm_hidden=64, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature,
            hidden_size=lstm_hidden,
            batch_first=True,            # 保持批次维度在前
            num_layers=2
        )

        # Multi-head attention  哪些时间步重要
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=4,
            batch_first=True
        )

        # 层归一化（LN）
        # self.layer_norm = nn.LayerNorm(lstm_hidden)

        # 特征门控/通道注意力层 (Feature Gating)    哪些 hidden 维重要
        # 目标是对最终的 lstm_hidden (128) 维度进行权重学习
        self.feature_gate = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 8), # Squeeze: 降维
            nn.ReLU(),
            nn.Linear(lstm_hidden // 8, lstm_hidden), 
            nn.Sigmoid() # Gating factor [0, 1]
        )
        
        # 全连接层    
        # self.fc = nn.Linear(lstm_hidden, 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        
        # CNN
        # permute: 只改变维度顺序，不改变数据存储
        # x = x.permute(0, 2, 1)       # → [batch, feature_dim, seq_len]
        # x = self.conv(x)       # → [batch, cnn_channels, seq_len]

        # x = self.pool(x)

        # LSTM 
        # x = x.permute(0, 2, 1)       # → [batch, seq_len, cnn_channels]
        lstm_output, _ = self.lstm(x)      # output -> [batch, seq_len, lstm_hidden]

        # Attention 重新“看一遍”整个序列
        attn_out, _ = self.attention(
            lstm_output,  # Q
            lstm_output,  # K
            lstm_output   # V
        )

        # 取最后一个时间步（信息汇总点）
        final_representation = attn_out[:, -1, :] # [batch, lstm_hidden]
        
        # Feature Gating (特征注意力)
        feature_weights = self.feature_gate(final_representation) # [batch, lstm_hidden]
        # 对最终的特征向量进行逐元素乘法加权
        gated_representation = final_representation * feature_weights 
        # 不重要的 hidden 维 → 被压小
        # 有用的 hidden 维 → 被保留

        # 全连接
        out = self.fc(gated_representation)

        # attn_out = self.layer_norm(attn_out + lstm_output)  # 残差连接

        # # 使用多层特征
        # last_hidden = h_n[-1]  # [batch, lstm_hidden]
        # attention_last = attn_out[:, -1, :]  # [batch, lstm_hidden]

        # # 合并特征
        # combined = torch.cat([attention_last, last_hidden], dim=-1)  # [batch, lstm_hidden*2]
        
        # # 全连接
        # out = self.fc(combined)

        # 最后一层 hidden state
        # last_hidden = h_n[-1]       # → [batch, lstm_hidden]
        # out = self.fc(last_hidden)  # → [batch, 1]

        return out