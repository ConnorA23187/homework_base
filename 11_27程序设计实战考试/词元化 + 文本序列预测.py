"""
词元化 + 文本序列预测
1. 构建分词
* 分词方式：按空格分词
* 最大token数量：2000
* 输出为int
* 输出序列长度自定
* 注意要把换行符替换成 <eos>, 此token表示文本结束

2. 构建序列预测数据集
- 从一个句子中构造多个训练样本，窗口大小自定, 注意标签只有一个token（预测下一个词）
- 示例：
- 输入：
- [我 喜欢 深度]
- 输出：
- [学习]

3. 构建模型

提示： 模型的结构包含：文本向量化，词嵌入，LSTM/GRU， Dense输出层

4. 训练模型

5. 根据输入的提示词，生成文本

输入前缀："深度 学习"， 要求模型输出预测下一个词

6. 实现循环文本生成（预测多个token）

"""

from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class SimpleTokenizer:
    def __init__(self, max_tokens=2000):
        self.max_tokens = max_tokens
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<eos>'}
        self.vocab_size = 3

    # 构建词典
    def build_vocab(self, text):
        # 将换行符替换为 <eos>
        text = text.replace('\n', ' <eos> ')
        # 以单词作为token基本单位
        tokens = text.split()
        counter = Counter(tokens)
        # 取最常见的 max_tokens - 3 个词
        most_common = counter.most_common(self.max_tokens - 3)
        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    # 词 转 id
    def encode(self, text):
        text = text.replace('\n', ' <eos> ')
        tokens = text.split()
        ids = []
        for token in tokens:
            ids.append(self.word2idx.get(token, self.word2idx['<unk>']))
        return ids

    # id 转词
    def decode(self, ids):
        return ' '.join([self.idx2word.get(i, '<unk>') for i in ids])


class TextDataset(Dataset):
    def __init__(self, tokenizer, text, seq_len=20):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        # 输入：从 idx 开始的 seq_len 个 token
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        # 下一个 token
        y = torch.tensor(self.tokens[idx + self.seq_len], dtype=torch.long)
        return x, y


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # 只取最后一个时间步用于预测下一个词
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


def train_model(model, dataloader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")


def predict_next_token(model, tokenizer, prefix, device='cpu'):
    model.eval()
    with torch.no_grad():
        ids = tokenizer.encode(prefix)
        x = torch.tensor([ids], dtype=torch.long).to(device)
        logits = model(x)
        pred_id = logits.argmax(dim=-1).item()
        return tokenizer.decode([pred_id])


def generate_text(model, tokenizer, prefix, max_new_tokens=5, seq_len=5, device='cpu'):
    model.eval()
    with torch.no_grad():
        ids = tokenizer.encode(prefix)
        # 预测词数量不足则填充
        if len(ids) < seq_len:
            ids = [tokenizer.word2idx['<pad>']] * (seq_len - len(ids)) + ids

        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-seq_len:]], dtype=torch.long).to(device)
            logits = model(x)
            pred_id = logits.argmax(dim=-1).item()
            if pred_id == tokenizer.word2idx['<eos>']:
                break
            ids.append(pred_id)

        return tokenizer.decode(ids)


# 数据准备
def get_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


text = get_data("data/corpus.txt")
# 分词器
tokenizer = SimpleTokenizer(max_tokens=2000)
tokenizer.build_vocab(text)
# 数据集
dataset = TextDataset(tokenizer, text, seq_len=10)
# DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# 初始化模型
model = LanguageModel(vocab_size=tokenizer.vocab_size, embed_dim=64, hidden_dim=128)

# 训练
train_model(model, dataloader, epochs=20, lr=0.005)

# 单步预测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
next_word = predict_next_token(model, tokenizer, "我 喜欢 深度", device=device)
print("预测下一个词:", next_word)

# 生成多词文本
generated = generate_text(model, tokenizer, "模型 训练", max_new_tokens=5, seq_len=3, device=device)
print("预测下一段词:", generated)
