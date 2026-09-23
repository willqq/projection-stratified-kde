import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =========================================
# 1. 单个自编码哈希模型
# =========================================
class HashAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, hash_bits=32):
        super(HashAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hash_bits)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hash_bits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.hash_bits = hash_bits

    def forward(self, x):
        h_real = self.encoder(x)              # 实值码
        h_sigmoid = torch.sigmoid(h_real)     # 映射到 [0,1]
        x_recon = self.decoder(h_sigmoid)     # 重建
        return h_real, h_sigmoid, x_recon

    def generate_hash(self, x):
        """推理时生成二值哈希码"""
        h_real = self.encoder(x)
        h_sigmoid = torch.sigmoid(h_real)
        return (h_sigmoid > 0.5).float()      # {0,1}^K


# =========================================
# 2. 损失函数
# =========================================
def hash_loss(x, x_recon, h_sigmoid, lambda_q=0.1, lambda_b=0.1):
    # (1) 重构损失
    L_rec = F.mse_loss(x_recon, x)

    # (2) 量化损失
    h_binary = (h_sigmoid > 0.5).float()
    L_quant = F.mse_loss(h_sigmoid, h_binary)

    # (3) 平衡损失
    bit_mean = torch.mean(h_sigmoid, dim=0)
    L_bal = torch.mean((bit_mean - 0.5) ** 2)

    return L_rec + lambda_q * L_quant #+ lambda_b * L_bal


# =========================================
# 3. 多模型集成（多个哈希表）
# =========================================
class MultiHashModels:
    def __init__(self, num_models, input_dim, hidden_dim=512, hash_bits=32, device="cpu"):
        self.models = [HashAutoencoder(input_dim, hidden_dim, hash_bits).to(device)
                       for _ in range(num_models)]
        self.device = device

    def train_all(self, data_loader, epochs=10, lr=1e-3):
        for idx, model in enumerate(self.models):
            print(f"=== 训练模型 {idx+1} ===")
            optimizer = optim.Adam(model.parameters(), lr=lr)
            for epoch in range(epochs):
                total_loss = 0
                for x in data_loader:
                    x = x.to(self.device)
                    h_real, h_sigmoid, x_recon = model(x)
                    loss = hash_loss(x, x_recon, h_sigmoid)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Model {idx+1}, Epoch {epoch+1}, Loss={total_loss/len(data_loader):.4f}")

    def build_hash_tables(self, dataset):
        """生成多个哈希表"""
        hash_tables = []
        with torch.no_grad():
            for model in self.models:
                hashes = model.generate_hash(dataset.to(self.device))
                hash_tables.append(hashes.cpu())
        return hash_tables


# =========================================
# 4. 检索函数
# =========================================
def hamming_distance(query, database):
    return torch.sum(torch.abs(query - database), dim=1)


def retrieve(query, hash_tables, top_k=10, mode="union"):
    """
    query: (1,K) 查询哈希码
    hash_tables: 多个表 [N,K]
    mode: "union"=并集增强召回 / "intersect"=交集提升精度
    """
    results = []
    for db in hash_tables:
        dist = hamming_distance(query, db)
        results.append(torch.argsort(dist)[:top_k])

    if mode == "union":
        return torch.unique(torch.cat(results))[:top_k]
    elif mode == "intersect":
        return torch.stack(results).unique_consecutive()  # 简化版交集
    else:
        return results[0]  # 默认只用第一个表


# =========================================
# 5. 示例运行
# =========================================
if __name__ == "__main__":
    # 模拟高维数据集
    input_dim = 1024
    dataset = torch.randn(500, input_dim)   # N=500, D=1024
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # 多模型哈希系统 (M=3)
    system = MultiHashModels(num_models=3, input_dim=input_dim, hash_bits=32, device="cpu")
    system.train_all(data_loader, epochs=1000, lr=1e-3)

    # 构建哈希表
    hash_tables = system.build_hash_tables(dataset)

    # 生成查询
    query = system.models[0].generate_hash(dataset[0:1])

    # 检索（并集模式）
    results = retrieve(query, hash_tables, top_k=5, mode="union")
    print("检索结果索引:", results.tolist())
