
import math
import time
import numpy as np
from numba import cuda
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
x_train = x_train.astype(np.float32)/255.0;  y_train = y_train.astype(np.int32)
x_test  = x_test.astype(np.float32)/255.0;   y_test  = y_test.astype(np.int32)
# pad
pad = (-len(x_train))%batch_size
if pad: x_train, y_train = np.vstack([x_train, x_train[:pad]]), np.hstack([y_train, y_train[:pad]])
pad = (-len(x_test))%batch_size
if pad: x_test, y_test   = np.vstack([x_test,  x_test[:pad]]),  np.hstack([y_test,  y_test[:pad]])
n_train, n_test = len(x_train), len(x_test)

# --- derived dims ---
H0, W0 = 28, 28
c1_h, c1_w = H0 - fsize + 1, W0 - fsize + 1
p1_h, p1_w = c1_h//2,      c1_w//2
c2_h, c2_w = p1_h - fsize + 1, p1_w - fsize + 1
p2_h, p2_w = c2_h//2,      c2_w//2
flat_size   = n_filters2 * p2_h * p2_w

# --- init weights + momentum buffers ---
Wc1 = np.random.randn(n_filters1,1,fsize,fsize).astype(np.float32)*0.1; bc1 = np.zeros(n_filters1, np.float32)
Wc2 = np.random.randn(n_filters2,n_filters1,fsize,fsize).astype(np.float32)*0.1; bc2 = np.zeros(n_filters2, np.float32)
W1  = np.random.randn(flat_size,hidden).astype(np.float32)*np.sqrt(2/flat_size); b1 = np.zeros(hidden, np.float32)
W2  = np.random.randn(hidden,10).astype(np.float32)*np.sqrt(2/hidden);           b2 = np.zeros(10,      np.float32)

vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)

# copy to GPU
d_Wc1, d_bc1 = cuda.to_device(Wc1), cuda.to_device(bc1)
d_Wc2, d_bc2 = cuda.to_device(Wc2), cuda.to_device(bc2)
d_W1,  d_b1  = cuda.to_device(W1),  cuda.to_device(b1)
d_W2,  d_b2  = cuda.to_device(W2),  cuda.to_device(b2)

# --- CUDA kernels ---

@cuda.jit
def conv2d_relu(inp, W, b, out):
    B, C, H, W_in = inp.shape
    F_out, _, f, _ = W.shape
    out_h, out_w = H-f+1, W_in-f+1
    bi = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    fo = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    idx= cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if bi<B and fo<F_out and idx<out_h*out_w:
        oy, ox = idx//out_w, idx%out_w
        acc = 0.0
        for ci in range(C):
            for u in range(f):
                for v in range(f):
                    acc += inp[bi,ci,oy+u,ox+v]*W[fo,ci,u,v]
        acc += b[fo]
        out[bi,fo,oy,ox] = acc if acc>0 else 0.0

@cuda.jit
def maxpool(inp, out):
    B,C,H,W_in = inp.shape
    oH,oW = H//2, W_in//2
    bi = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ci = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    idx= cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if bi<B and ci<C and idx<oH*oW:
        oy, ox = idx//oW, idx%oW
        v = inp[bi,ci,oy*2,ox*2]
        t = inp[bi,ci,oy*2,ox*2+1]; v = t if t>v else v
        t = inp[bi,ci,oy*2+1,ox*2];   v = t if t>v else v
        t = inp[bi,ci,oy*2+1,ox*2+1]; v = t if t>v else v
        out[bi,ci,oy,ox] = v

@cuda.jit
def dense_relu(inp, W, b, out):
    i,j = cuda.grid(2)
    if i<inp.shape[0] and j<W.shape[1]:
        acc = b[j]
        for k in range(inp.shape[1]):
            acc += inp[i,k]*W[k,j]
        out[i,j] = acc if acc>0 else 0.0

@cuda.jit
def dense_softmax(inp, W, b, logits, probs):
    i,j = cuda.grid(2)
    if i<inp.shape[0] and j<W.shape[1]:
        tmp = b[j]
        for k in range(inp.shape[1]):
            tmp += inp[i,k]*W[k,j]
        logits[i,j] = tmp
    cuda.syncthreads()
    if i<probs.shape[0]:
        mx = -1e9
        for t in range(probs.shape[1]):
            if logits[i,t]>mx: mx=logits[i,t]
        s=0.0
        for t in range(probs.shape[1]):
            s += math.exp(logits[i,t]-mx)
        for t in range(probs.shape[1]):
            probs[i,t] = math.exp(logits[i,t]-mx)/s

@cuda.jit
def backprop_output(probs, y, dlogits):
    i = cuda.grid(1)
    if i<y.shape[0]:
        for j in range(probs.shape[1]):
            dlogits[i,j] = probs[i,j] - (j==y[i])

# --- pre-alloc buffers ---
d_conv1 = cuda.device_array((batch_size,n_filters1,c1_h,c1_w),np.float32)
d_pool1 = cuda.device_array((batch_size,n_filters1,p1_h,p1_w),np.float32)
d_conv2 = cuda.device_array((batch_size,n_filters2,c2_h,c2_w),np.float32)
d_pool2 = cuda.device_array((batch_size,n_filters2,p2_h,p2_w),np.float32)
d_hid   = cuda.device_array((batch_size,hidden),         np.float32)
d_logits= cuda.device_array((batch_size,10),             np.float32)
d_probs = cuda.device_array((batch_size,10),             np.float32)
d_dlog  = cuda.device_array((batch_size,10),             np.float32)

# --- launch configs ---
threads_conv = (16,8,4)   # 512 threads/block
blocks_c1    = ((batch_size+15)//16, (n_filters1+7)//8, (c1_h*c1_w+3)//4)
blocks_c2    = ((batch_size+15)//16, (n_filters2+7)//8, (c2_h*c2_w+3)//4)
threads_pool = threads_conv
blocks_p1    = ((batch_size+15)//16, (n_filters1+7)//8, (p1_h*p1_w+3)//4)
blocks_p2    = ((batch_size+15)//16, (n_filters2+7)//8, (p2_h*p2_w+3)//4)
threads2d    = (16,16)
blocks_hid   = ((batch_size+15)//16,(hidden+15)//16)
blocks_out   = ((batch_size+15)//16,(10+15)//16)
grid_dlog    = (batch_size+255)//256; threads_dlog=256

# --- timing and training & test loops ---
start_total = time.time()
for ep in range(epochs):
    epoch_start = time.time()

    train_corr, train_loss = 0, 0.0
    for bs in range(0,n_train,batch_size):
        xb, yb = x_train[bs:bs+batch_size], y_train[bs:bs+batch_size]
        d_xb = cuda.to_device(xb.reshape(batch_size,1,28,28))
        d_yb = cuda.to_device(yb)

        conv2d_relu[blocks_c1,threads_conv](d_xb,     d_Wc1,d_bc1,d_conv1)
        maxpool     [blocks_p1,threads_pool](d_conv1,d_pool1)
        conv2d_relu[blocks_c2,threads_conv](d_pool1,   d_Wc2,d_bc2,d_conv2)
        maxpool     [blocks_p2,threads_pool](d_conv2,d_pool2)

        d_flat = d_pool2.reshape(batch_size,flat_size)
        dense_relu  [blocks_hid,threads2d](d_flat, d_W1,d_b1,d_hid)
        dense_softmax[blocks_out,threads2d](d_hid,  d_W2,d_b2,d_logits,d_probs)

        # training metrics
        probs = d_probs.copy_to_host()
        losses= -np.log(probs[np.arange(batch_size),yb]+1e-8)
        train_loss += losses.sum()
        preds = np.argmax(probs,axis=1)
        train_corr += (preds==yb).sum()

        # FC-backprop + host momentum update
        backprop_output[grid_dlog,threads_dlog](d_probs,d_yb,d_dlog)
        dlog_h = d_dlog.copy_to_host(); hid_h = d_hid.copy_to_host(); flat_h = d_flat.copy_to_host()

        grad_W2 = hid_h.T.dot(dlog_h); grad_b2 = dlog_h.sum(axis=0)
        dh = dlog_h.dot(W2.T); dh[hid_h<=0]=0
        grad_W1 = flat_h.T.dot(dh);      grad_b1 = dh.sum(axis=0)

        vW2[:] = momentum*vW2 + (1-momentum)*grad_W2; vb2[:] = momentum*vb2 + (1-momentum)*grad_b2
        W2  -= lr*vW2; b2  -= lr*vb2
        vW1[:] = momentum*vW1 + (1-momentum)*grad_W1; vb1[:] = momentum*vb1 + (1-momentum)*grad_b1
        W1  -= lr*vW1; b1  -= lr*vb1

        d_W2.copy_to_device(W2); d_b2.copy_to_device(b2)
        d_W1.copy_to_device(W1); d_b1.copy_to_device(b1)

    train_time = time.time() - epoch_start

    # test
    test_corr = 0
    for ts in range(0,n_test,batch_size):
        xt, yt = x_test[ts:ts+batch_size], y_test[ts:ts+batch_size]
        d_xt = cuda.to_device(xt.reshape(batch_size,1,28,28))

        conv2d_relu[blocks_c1,threads_conv](d_xt,     d_Wc1,d_bc1,d_conv1)
        maxpool     [blocks_p1,threads_pool](d_conv1,d_pool1)
        conv2d_relu[blocks_c2,threads_conv](d_pool1,   d_Wc2,d_bc2,d_conv2)
        maxpool     [blocks_p2,threads_pool](d_conv2,d_pool2)

        d_flat_t = d_pool2.reshape(batch_size,flat_size)
        dense_relu  [blocks_hid,threads2d](d_flat_t, d_W1,d_b1,d_hid)
        dense_softmax[blocks_out,threads2d](d_hid,  d_W2,d_b2,d_logits,d_probs)

        preds_t = np.argmax(d_probs.copy_to_host(),axis=1)
        test_corr += (preds_t==yt).sum()

    print(f"Epoch {ep+1}/{epochs}  "
          f"train acc {train_corr/n_train:.4f}, "
          f"test acc {test_corr/n_test:.4f}, "
          f"time {train_time:.2f}s")

total_time = time.time() - start_total
print(f"Total training time: {total_time:.2f}s")
