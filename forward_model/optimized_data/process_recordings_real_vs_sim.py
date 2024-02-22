#%%
import numpy as np
import matplotlib.pyplot as plt

#real = np.load('real_sines_80_mics.npy')
#real = np.load('EARS/forward_model/optimized_data/Jan10_23-48-42_lap983_Jan10_11-26-20_aida_2_rads_80mics_128sources_051_real_sines.npy')
#real =  np.load('EARS/forward_model/optimized_data/New folder1/Jan10_23-48-42_lap983_Jan10_11-26-20_aida_2_rads_80mics_128sources_051_real_rec.npy')
real = np.load('EARS/forward_model/optimized_data/all_dist/real_rec.npy')
#pytsim = np.load('simulated_sines_051.npy')
#sim = np.load('EARS/forward_model/optimized_data/Jan10_23-48-42_lap983_Jan10_11-26-20_aida_2_rads_80mics_128sources_051_simulated_rec.npy')
#sim = np.load('EARS/forward_model/optimized_data/New folder1/Jan10_23-48-42_lap983_Jan10_11-26-20_aida_2_rads_80mics_128sources_051_real_sines.npy')
sim = np.load('EARS/forward_model/optimized_data/all_dist/simulated_rec.npy')
vel = 23.46668440418565
fs = 3003.735603735763
mics_range = [53, 57, 63, 68, 73, 83, 93, 103]
max_all = max(real[0].max(), sim[0].max())
min_all = min(real[0].min(), sim[0].min())

#subplots_labels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$']
subplots_labels = [r'$0$', r'$\pi$ \ 2', r'$\pi$', r'$3\pi$ \ 2']

# %%

fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True, figsize=(7, 7))
plt.subplots_adjust(hspace = 0.3, wspace = 0.1, left=0.15)
total_error = np.zeros((4, 8))
for j in range(8):
    ax = axes[j//2, j%2]

    for i in range(4):
        # is 20 the number of samples between quarters of pi?
        # is 80 the number of samples between a distance and another one? (since it's 4 times 20 it seems to be the case)
        # Hence they use 80 virtual microphones to simulate the 8 real ones!
        cur_real = real[i*20 + j*80][450:740]
        ax.plot(cur_real, label=f'real {subplots_labels[i]}', lw=0.8)

        cur_sim = sim[i*20 + j*80][450:740]
        ax.plot(cur_sim, label=f'sim {subplots_labels[i]}', lw=0.8)

        ax.set_title(f'Distance {mics_range[j]/100} [m]', size=10)
        total_error[i, j] = np.linalg.norm(cur_real - cur_sim)
        print(f"total error {mics_range[j]} {subplots_labels[i]}: {np.linalg.norm(cur_real - cur_sim)}")
        #print(f"shown error {mics_range[j]} {subplots_labels[i]}: {np.linalg.norm(cur_real[450:740] - cur_sim[450:740])}")
        if j==7:
            lgd = ax.legend(loc='lower center', bbox_to_anchor=(-0.04, -1.2),
                             ncol=4) #, fancybox=True, shadow=True)

plt.ylim(min_all, max_all)
# plt.legend(loc='lower left', mode="expand", ncol=4)
ticks = [str(round(tick / fs,2)) for tick in  plt.xticks()[0] ]
plt.xticks(plt.xticks()[0][1:-1], ticks[1:-1])

fig.text(0.5, 0.04, 'Time [sec]', ha='center')
fig.text(0.04, 0.5, 'Signed magnitude', va='center', rotation='vertical')


# plt.suptitle(f'Ground truth vs. simulated recordings, distance {mics_range[j]/100} [m]')
plt.savefig(f'all_dist_transformer.pdf', bbox_inches='tight')



#plt.show()

plt.clf()
# plt.plot(total_error)
# plt.savefig("t_transformer_total_error.pdf", bbox_inches="tight")
# %%
