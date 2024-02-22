import numpy as np
import matplotlib.pyplot as plt

# import matplotlib
# print(matplotlib.matplotlib_fname())

# import matplotlib.rcsetup as rcsetup
# print(rcsetup.all_backends)

# matplotlib.use("Qt5Agg")

sim = np.load("./simulated_rec.npy")
print(f"shape: {sim.shape}")
plt.plot(sim)
plt.savefig("sim_sines.png", format="png")

# real = np.load("./real_sines.npy")
# print(f"shape: {real.shape}")
# plt.plot(real)
# plt.savefig("real_sines.png", format="png")

# enc = np.load("./encoder_readings.npy")
# print(f"shape: {enc.shape}")
# plt.plot(enc)
# plt.savefig("encoder.png")