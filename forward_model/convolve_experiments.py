import jax.numpy as jnp
from jax import lax
from jax.scipy import signal
from jax import random
from jax import grad, jit
from time import time
import os

def convolve(h, sig):
    x = signal.convolve(h, sig)
    return x

conv_jit = jit(convolve)

def using_conv(hs, sigs, num_hs, num_sigs):
    for hi in range(num_hs):
        for sigi in range(num_sigs):
            convolve(hs[hi], sigs[sigi])

def using_conv_jit(hs, sigs, num_hs, num_sigs):
    for hi in range(num_hs):
        for sigi in range(num_sigs):
            conv_jit(hs[hi], sigs[sigi])


os.environ['CUDA_VISIBLE_DEVICES']= '2'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

key = random.PRNGKey(0)
key, subkey = random.split(key)

h = random.uniform(key, (40,))
sig = random.uniform(subkey, (50,))
# start_time = time()
# x = convolve(h, sig)
# print(f'{time() - start_time}')

x = jnp.array([1, 2, 1, 5, 6,7,8]).astype(float)
y = jnp.array([4, 5, 1, 1, 2]).astype(float)
y2 = jnp.array([5, 6, 1, 3, 1]).astype(float)
res = jnp.convolve(x, y)
res2 = jnp.convolve(x, y2)


# using lax.conv_general_dilated to implement jnp.convolve
x_jax = jnp.reshape(x,(1,1,len(x)))
y_jax = jnp.flip(jnp.reshape(y,(1,1,len(y))),2) #flip rhs

res1 = lax.conv_general_dilated(
    x_jax, 
    y_jax,
    window_strides=(1,),
    padding=[(len(y) - 1, len(y) - 1)])  # equivalent of padding='full' in NumPy

print(jnp.all(res1== res))

# y_all = jnp.vstack([y, y2])
# y_all_flipped = jnp.flip(jnp.reshape(y_all, (y_all.shape[0], 1, y_all.shape[1])),2)

# # fucking batch
# res_all = lax.conv_general_dilated(
#     x_jax,
#     y_all_flipped,
#     window_strides=(1,),
#     padding=[(y_all.shape[1] - 1, y_all.shape[1] - 1)])  # equivalent of padding='full' in NumPy

# print(jnp.all(res_all[0][0] == res))
# print(jnp.all(res_all[0][1] == res2))


def batch_convolve1d(x, y):
    # x: 1d array
    # y: batch of N 1d arrays
    # out: Nx(np.convolve(x,y[i]).shape[0])
    x_reshaped = jnp.reshape(x,(1,1,len(x)))
    y_flipped = jnp.flip(jnp.reshape(y, (y.shape[0], 1, y.shape[1])),2)

    # fucking batch
    res = lax.conv_general_dilated(
        x_reshaped,
        y_flipped,
        window_strides=(1,),
        padding=[(y.shape[1] - 1, y.shape[1] - 1)])  # equivalent of padding='full' in NumPy
    
    return res

y_all = jnp.vstack([y, y2])
res_all = batch_convolve1d(x, y_all)

print(jnp.all(res_all[0][0] == res))
print(jnp.all(res_all[0][1] == res2))


