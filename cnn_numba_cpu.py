# cnn_cpu_numba.py

import time
import math
import numpy as np
from numba import njit, prange
from tensorflow.keras.datasets import mnist

# --- hyperparameters & momentum ---
lr          = 0.01
momentum    = 0.9
batch_size  = 512
epochs      = 10
n_filters1  = 8
n_filters2  = 16
fsize       = 3
hidden      = 128

# --- load & preprocess MNIST ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32)/255.0
y_train = y_train.astype(np.int32)
x_test  = x_test.astype(np.float32)/255.0
y_test  = y_test.astype(np.int32)

# pad to full batches
pad_train = (-len(x_train)) % batch_size
if pad_train:
    x_train = np.vstack([x_train, x_train[:pad_train]])
    y_train = np.hstack([y_train, y_train[:pad_train]])
pad_test = (-len(x_test)) % batch_size
if pad_test:
    x_test  = np.vstack([x_test,  x_test[:pad_test]])
    y_test  = np.hstack([y_test,  y_test[:pad_test]])
n_train, n_test = len(x_train), len(x_test)

# --- derived dims ---
H0, W0 = 28, 28
c1_h, c1_w = H0 - fsize + 1, W0 - fsize + 1
p1_h, p1_w = c1_h//2,      c1_w//2
c2_h, c2_w = p1_h - fsize + 1, p1_w - fsize + 1
p2_h, p2_w = c2_h//2,      c2_w//2
flat_size   = n_filters2 * p2_h * p2_w

# --- init weights + momentum buffers ---
rng = np.random.RandomState(123)
Wc1 = rng.randn(n_filters1,1,fsize,fsize).astype(np.float32)*0.1; bc1 = np.zeros(n_filters1, np.float32)
Wc2 = rng.randn(n_filters2,n_filters1,fsize,fsize).astype(np.float32)*0.1; bc2 = np.zeros(n_filters2, np.float32)
W1  = rng.randn(flat_size,hidden).astype(np.float32)*np.sqrt(2/flat_size); b1 = np.zeros(hidden, np.float32)
W2  = rng.randn(hidden,10).astype(np.float32)*np.sqrt(2/hidden);           b2 = np.zeros(10,      np.float32)
vWc1 = np.zeros_like(Wc1);  vbc1 = np.zeros_like(bc1)
vWc2 = np.zeros_like(Wc2);  vbc2 = np.zeros_like(bc2)
vW1  = np.zeros_like(W1);   vb1  = np.zeros_like(b1)
vW2  = np.zeros_like(W2);   vb2  = np.zeros_like(b2)

# --- CPU‐JIT kernels ---
@njit(parallel=True, fastmath=True)
def conv2d_relu_cpu(inp, W, b, out):
    B, C, H, W_in = inp.shape
    F, _, f, _ = W.shape
    oh, ow = H-f+1, W_in-f+1
    for bi in prange(B):
        for fo in range(F):
            for oy in range(oh):
                for ox in range(ow):
                    acc = b[fo]
                    for ci in range(C):
                        for u in range(f):
                            for v in range(f):
                                acc += inp[bi,ci,oy+u,ox+v] * W[fo,ci,u,v]
                    out[bi,fo,oy,ox] = acc if acc>0 else 0.0

@njit(parallel=True, fastmath=True)
def maxpool_cpu(inp, out):
    B,C,H,W_in = inp.shape
    oh, ow = H//2, W_in//2
    for bi in prange(B):
        for ci in range(C):
            for oy in range(oh):
                for ox in range(ow):
                    i0 = inp[bi,ci,oy*2,  ox*2]
                    i1 = inp[bi,ci,oy*2,  ox*2+1]
                    i2 = inp[bi,ci,oy*2+1,ox*2]
                    i3 = inp[bi,ci,oy*2+1,ox*2+1]
                    m = i0 if i0>i1 else i1
                    m = i2 if i2>m  else m
                    out[bi,ci,oy,ox] = i3 if i3>m  else m

@njit(parallel=True, fastmath=True)
def dense_relu_cpu(inp, W, b, out):
    B,D = inp.shape
    _, K = W.shape
    for i in prange(B):
        for j in range(K):
            acc = b[j]
            for k in range(D):
                acc += inp[i,k]*W[k,j]
            out[i,j] = acc if acc>0 else 0.0

@njit(parallel=True, fastmath=True)
def dense_softmax_cpu(inp, W, b, logits, probs):
    B,D = inp.shape
    _, K = W.shape
    for i in prange(B):
        # logits
        for j in range(K):
            acc = b[j]
            for k in range(D):
                acc += inp[i,k]*W[k,j]
            logits[i,j] = acc
        # softmax
        mx = logits[i,0]
        for j in range(1,K):
            if logits[i,j]>mx: mx = logits[i,j]
        s = 0.0
        for j in range(K):
            ex = math.exp(logits[i,j]-mx)
            probs[i,j] = ex
            s += ex
        for j in range(K):
            probs[i,j] /= s

@njit(parallel=True)
def backprop_output_cpu(probs, y, dlogits):
    B,K = probs.shape
    for i in prange(B):
        yi = y[i]
        for j in range(K):
            dlogits[i,j] = probs[i,j] - (j==yi)

# --- pre‑allocate buffers ---
conv1   = np.empty((batch_size,n_filters1,c1_h,c1_w), np.float32)
pool1   = np.empty((batch_size,n_filters1,p1_h,p1_w), np.float32)
conv2   = np.empty((batch_size,n_filters2,c2_h,c2_w), np.float32)
pool2   = np.empty((batch_size,n_filters2,p2_h,p2_w), np.float32)
hid_out = np.empty((batch_size,hidden),            np.float32)
logits  = np.empty((batch_size,10),                np.float32)
probs   = np.empty((batch_size,10),                np.float32)
dlogits = np.empty((batch_size,10),                np.float32)

# --- training & evaluation ---
t0 = time.time()
for ep in range(epochs):
    epoch_start = time.time()
    train_corr, train_loss = 0, 0.0
    for bs in range(0, n_train, batch_size):
        xb = x_train[bs:bs+batch_size].reshape(batch_size,1,28,28)
        yb = y_train[bs:bs+batch_size]

        conv2d_relu_cpu(xb,    Wc1, bc1, conv1)
        maxpool_cpu    (conv1,       pool1)
        conv2d_relu_cpu(pool1, Wc2, bc2, conv2)
        maxpool_cpu    (conv2,       pool2)

        flat = pool2.reshape(batch_size, flat_size)
        dense_relu_cpu(flat, W1, b1, hid_out)
        dense_softmax_cpu(hid_out, W2, b2, logits, probs)

        loss_batch = -np.log(probs[np.arange(batch_size), yb] + 1e-8)
        train_loss += loss_batch.sum()
        preds = np.argmax(probs, axis=1)
        train_corr += (preds==yb).sum()

        backprop_output_cpu(probs, yb, dlogits)
        grad_W2 = hid_out.T.dot(dlogits);  grad_b2 = dlogits.sum(axis=0)
        dh = dlogits.dot(W2.T); dh[hid_out<=0] = 0
        grad_W1 = flat.T.dot(dh);          grad_b1 = dh.sum(axis=0)

        vW2 = momentum*vW2 + (1-momentum)*grad_W2;  W2 -= lr*vW2
        vb2 = momentum*vb2 + (1-momentum)*grad_b2;  b2 -= lr*vb2
        vW1 = momentum*vW1 + (1-momentum)*grad_W1;  W1 -= lr*vW1
        vb1 = momentum*vb1 + (1-momentum)*grad_b1;  b1 -= lr*vb1

    test_corr = 0
    for bs in range(0, n_test, batch_size):
        xb = x_test[bs:bs+batch_size].reshape(batch_size,1,28,28)
        yb = y_test[bs:bs+batch_size]
        conv2d_relu_cpu(xb,    Wc1, bc1, conv1)
        maxpool_cpu    (conv1,       pool1)
        conv2d_relu_cpu(pool1, Wc2, bc2, conv2)
        maxpool_cpu    (conv2,       pool2)
        flat = pool2.reshape(batch_size, flat_size)
        dense_relu_cpu(flat, W1, b1, hid_out)
        dense_softmax_cpu(hid_out, W2, b2, logits, probs)
        preds = np.argmax(probs, axis=1)
        test_corr += (preds==yb).sum()

    print(f"Epoch {ep+1}/{epochs}  train acc {train_corr/n_train:.4f}, test acc {test_corr/n_test:.4f}, Time: {time.time()-epoch_start:.2f}")
    # print(f"Epoch {ep+1} time: {time.time()-epoch_start:.2f}s")

print(f"Total time: {time.time()-t0:.2f}s")
print("Done.")

#mprof run python cpu_impl.py
#mprof plot
