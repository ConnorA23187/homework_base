import struct
import numpy as np
import gzip

def add(x, y):
    """一个简单的add函数，以便熟悉自动测试（pytest）

    Args:
        x (Python数字 或者 numpy array)
        y (Python数字 或者 numpy array)

    Return:
        x+y的和
    """
    return x + y


def parse_mnist(image_filename, label_filename):
    """ 读取 MNIST 格式的图像和标签文件。有关文件格式的说明，请参阅此页面：
    http://yann.lecun.com/exdb/mnist/。

    参数：
    image_filename（字符串）：MNIST 格式的 gzip 压缩图像文件的名称
    label_filename（字符串）：MNIST 格式的 gzip 压缩标签文件的名称

    返回：
    tuple (X,y)：
    x (numpy.ndarray[np.float32])：包含已加载数据的二维 numpy 数组。数据的维度应为
    (num_examples x input_dim)，其中“input_dim”是数据的完整维度，例如，由于 MNIST 图像为 28x28，因此
    input_dim 为 784。值应为 np.float32 类型，并且数据应被归一化为最小值为 0.0，
    最大值为 1.0 （即将原始值 0 缩放为 0.0，将 255 缩放为 1.0）。

    y (numpy.ndarray[dtype=np.uint8])：包含示例标签的一维 NumPy 数组。值应为 np.uint8 类型，对于 MNIST，将包含 0-9 的值。
    """

    with gzip.open(image_filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"图像文件魔法数错误: {hex(magic)}")
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows * cols).astype(np.float32) / 255.0  # 展平为 (num_examples, 784)

    # 读取标签文件
    with gzip.open(label_filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"标签文件魔法数错误: {hex(magic)}")

        labels = np.frombuffer(f.read(), dtype=np.uint8)  # 保持为整数

    if num_images != num_labels:
        raise ValueError("图像和标签数量不匹配")

    return images, labels


def softmax_loss(Z, y):
    """ 返回 softmax 损失。

    参数：
    z (np.ndarray[np.float32])：形状为 (batch_size, num_classes) 的二维 NumPy 数组，
    包含每个类别的 对数概率 预测值 （softmax函数激活之前的值）。

    y (np.ndarray[np.uint8])：形状为 (batch_size, ) 的一维 NumPy 数组，包含每个样本的真实标签。

    返回：
    样本的平均 softmax 损失。
    """

    batch_size = Z.shape[0]
    num_classes = Z.shape[1]

    # 防止数值不稳定：减去每行的最大值（避免指数爆炸）
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    Z_shifted = np.clip(Z_shifted, -88.7, 88.7)

    # 计算 softmax 概率
    exp_z = np.exp(Z_shifted)
    softmax_probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # 提取真实标签对应的概率（logits 模式）
    probs_correct_class = softmax_probs[np.arange(batch_size), y.flatten()]

    # 计算交叉熵损失（取负对数）
    loss = -np.mean(np.log(probs_correct_class + 1e-15))  # 加 1e-15 防止 log(0)

    return loss


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ 使用步长 lr 和指定的批次大小，对数据运行单轮 小批量梯度下降 进行 softmax 回归。此函数会修改
    θ 矩阵，并迭代 X 中的批次，但不对顺序进行随机化。

    参数：
    X (np.ndarray[np.float32])：大小为
    (num_examples x input_dim) 的二维输入数组。
    y (np.ndarray[np.uint8])：大小为 (num_examples,) 的一维类别标签数组。
    theta (np.ndarrray[np.float32])：softmax 回归的二维数组参数，形状为 (input_dim, num_classes)。
    lr (float)：SGD 的步长（学习率）。
    batch (int)：SGD 小批次的大小。

    返回：
    无
    """
    num_examples = X.shape[0]
    input_dim, num_classes = theta.shape

    # 按批次遍历数据（不随机打乱顺序）
    for i in range(0, num_examples, batch):
        # 获取当前批次的样本和标签
        X_batch = X[i: i + batch]
        y_batch = y[i: i + batch]
        batch_actual_size = X_batch.shape[0]  # 处理最后一个不完整的批次

        logits = X_batch @ theta
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)

        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # 形状 (batch_actual_size, num_classes)

        y_onehot = np.zeros((batch_actual_size, num_classes), dtype=np.float32)
        y_onehot[np.arange(batch_actual_size), y_batch] = 1.0

        # 计算梯度
        gradient = X_batch.T @ (probs - y_onehot) / batch_actual_size

        theta -= lr * gradient

def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ 对由权重 W1 和 W2 定义的双层神经网络（无偏差项）运行一个 小批量梯度下降 迭代轮次：
    logits = ReLU(X * W1) * W2
    该函数应使用步长 lr 和指定的批次大小（并且同样，不随机化 X 的顺序）。它应修改 W1 和 W2 矩阵。

    参数：
    X (np.ndarray[np.float32])：大小为 (num_examples x input_dim) 的二维输入数组。
    y (np.ndarray[np.uint8])：大小为 (num_examples,) 的一维类别标签数组。
    W1 (np.ndarray[np.float32])：第一层权重的二维数组，形状为(input_dim, hidden_dim)
    W2 (np.ndarray[np.float32])：第二层权重的二维数组。形状
    (hidden_dim, num_classes)
    lr (float)：SGD 的步长（学习率）
    batch (int)：SGD 小批次的大小

    返回：
    无
    """

    num_examples = X.shape[0]
    _, num_classes = W2.shape

    # 按批次遍历数据（不随机打乱顺序）
    for i in range(0, num_examples, batch):
        # 获取当前批次的样本和标签
        X_batch = X[i: i + batch]
        y_batch = y[i: i + batch]
        batch_actual_size = X_batch.shape[0]  # 处理最后一个不完整的批次

        # 前向传播
        hidden_layer = np.maximum(0, X_batch @ W1)
        logits = hidden_layer @ W2

        # Softmax
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        y_onehot = np.zeros((batch_actual_size, num_classes), dtype=np.float32)
        y_onehot[np.arange(batch_actual_size), y_batch] = 1.0

        # 反向传播
        grad_W2 = hidden_layer.T @ (probs - y_onehot) / batch_actual_size

        # 隐藏层的梯度
        grad_hidden = (probs - y_onehot) @ W2.T

        # ReLU 的梯度
        relu_mask = (hidden_layer > 0).astype(np.float32)

        # W1 的梯度
        grad_W1 = X_batch.T @ (grad_hidden * relu_mask) / batch_actual_size

        # 更新参数
        W1 -= lr * grad_W1
        W2 -= lr * grad_W2



### 下面的代码不用编辑，只是用来展示功能的

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100):
    """ 示例函数，用softmax回归训练 """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ 示例函数，训练神经网络 """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "../data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "../data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)