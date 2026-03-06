import numpy as np
import os
sizes=[]

np.random.seed(1337)

def init_params(sizes):
    W, b = [], []
    for i in range(len(sizes)-1):
        insz, outsz = sizes[i], sizes[i+1]
        W.append((0.01 * np.random.randn(insz, outsz)).astype(np.float32))
        b.append(np.zeros((1, outsz), np.float32))
    return W, b

def relu(x):
    return np.maximum(x, 0.0)

def reluprim(Z):
    return (Z>0).astype(np.float32)

def softmax(vals):
    z = vals - vals.max(axis=1, keepdims=True)
    e = np.exp(z, dtype=np.float32)
    return e / e.sum(axis=1, keepdims=True)

def forward(X, W, b):
    A = X
    As = [A]
    Zs = []
    L = len(W)
    for i in range(L):
        Z = A @ W[i] + b[i]
        Zs.append(Z)
        if i < L-1:
            A = relu(Z)
        else:
            A = Z
        As.append(A)
    return A, (As, Zs)

def loss(vals, y, W):
    z = vals - vals.max(axis=1, keepdims=True)
    expz = np.exp(z, dtype=np.float32)
    prob = expz / expz.sum(axis=1, keepdims=True)
    B = prob.shape[0]
    p_correct = prob[np.arange(B), y]
    ce = -np.mean(np.log(p_correct + 1e-12))
    return float(ce), prob

def backprop(prob,y, W, As, Zs):
    B = prob.shape[0]
    L = len(W)
    dZ = prob.copy()
    dZ[np.arange(B), y] -= 1.0
    dZ /= B
    dW = [None] * L
    db = [None] * L
    for i in reversed(range(L)):
        A_prev = As[i]
        dW[i] = A_prev.T @ dZ
        db[i] = dZ.sum(axis=0, keepdims=True)
        if i != 0:
            dA_prev = dZ @ W[i].T
            dZ = dA_prev * reluprim(Zs[i - 1])
    return dW, db

def predict(X, W, b):
    vals, skip = forward(X, W, b)
    return vals.argmax(axis=1)

def sgd(W, b, dW, db, lr):
    for i in range(len(W)):
        W[i] -= lr * dW[i]
        b[i] -= lr * db[i]

def train(Xtr, ytr, Xval, yval, sizes, lr=1e-1, epochs=5, batch_size=128):
    W, b = init_params(sizes)
    N=Xtr.shape[0]
    for ep in range(1, epochs+1):
        idx = np.random.permutation(N)
        shuffledX=Xtr[idx]
        shuffledY=ytr[idx]
        for i in range(0, N, batch_size):
            batchX=shuffledX[i:i+batch_size]
            batchY=shuffledY[i:i+batch_size]
            vals, (As, Zs)=forward(batchX, W, b)
            L, prob = loss(vals, batchY, W)
            dW, db=backprop(prob, batchY, W, As, Zs)
            sgd(W,b, dW, db, lr)
    return W, b

def accuracy(X, y, W, b):
    return float((predict(X, W, b) == y).mean())

def read_idx_images(path):
    raw = np.fromfile(path, dtype=np.uint8)
    magic, num, rows, cols = raw[:16].view('>u4')
    X = raw[16:].reshape(num, rows * cols).astype(np.float32) / 255.0
    return X

def read_idx_labels(path):
    raw = np.fromfile(path, dtype=np.uint8)
    magic, num = raw[:8].view('>u4')
    y = raw[8:].astype(np.int64)
    return y

def load_mnist_idx(train_images_path, train_labels_path, test_images_path, test_labels_path):
    Xtr = read_idx_images(train_images_path)
    ytr = read_idx_labels(train_labels_path)
    Xte = read_idx_images(test_images_path)
    yte = read_idx_labels(test_labels_path)
    return Xtr, ytr, Xte, yte

train_images_path = "train-images.idx3-ubyte"
train_labels_path = "train-labels.idx1-ubyte"
test_images_path  = "t10k-images.idx3-ubyte"
test_labels_path  = "t10k-labels.idx1-ubyte"

Xtr, ytr, Xte, yte = load_mnist_idx(train_images_path, train_labels_path, test_images_path, test_labels_path)

sizes = [784, 128, 10]

W, b = train(Xtr, ytr, Xte, yte, sizes, lr=0.01, epochs=50, batch_size=32)

test_acc = accuracy(Xte, yte, W, b)
print(f"Test accuracy: {test_acc:.4f}")
