import statistics
import matplotlib.pyplot as plt # type: ignore

data_mpc_s1 = [154, 173, 88, 144, 147, 107, 139, 164, 88, 145] # fail: 2
data_ddpg_s1 = [120, 126, 117, 118, 116, 117, 116, 120, 114, 120] # fail: 0
data_hyb_s1 = [53, 51, 51, 51, 51, 53, 51, 53, 52, 51] # fail: 0

data_mpc_s2 = [96, 101, 110, 123, 136, 145, 149, 167, 199, 160] # fail: 0
data_ddpg_s2 = [151, 128, 135, 170, 153, 150, 143, 125, 128, 178] # fail: 0
data_hyb_s2 = [58, 59, 61, 63, 58, 58, 62, 59, 59, 63] # fail: 0

data_mpc_s3 = [75, 75, 113, 72, 123, 71, 72, 71, 70, 70] # fail: 1
data_ddpg_s3 = [142, 143, 155, 153, 161, 153, 142, 146, 142, 155] # fail: 0
data_hyb_s3 = [85, 98, 86, 132, 92, 134, 147, 118, 91, 92] # fail: 1

n_bar_per_category = 3
bar_width = 0.5
yerr_width = 5
color_list = ['#2878b5', '#9ac9db', '#f8ac8c']

fig, ax = plt.subplots()

mpc_index = [i*(n_bar_per_category+1)*bar_width for i in range(n_bar_per_category)]
ddpg_index = [bar_width + i*(n_bar_per_category+1)*bar_width for i in range(n_bar_per_category)]
hyb_index = [2*bar_width + i*(n_bar_per_category+1)*bar_width for i in range(n_bar_per_category)]

# add the mean and error bars
bar_mpc = ax.bar(mpc_index, 
       [statistics.mean(data_mpc_s1), statistics.mean(data_mpc_s2), statistics.mean(data_mpc_s3)], bar_width, color=color_list[0],
       yerr=[statistics.stdev(data_mpc_s1), statistics.stdev(data_mpc_s2), statistics.stdev(data_mpc_s3)], capsize=yerr_width)
bar_ddpg = ax.bar(ddpg_index,
       [statistics.mean(data_ddpg_s1), statistics.mean(data_ddpg_s2), statistics.mean(data_ddpg_s3)], bar_width, color=color_list[1],
       yerr=[statistics.stdev(data_ddpg_s1), statistics.stdev(data_ddpg_s2), statistics.stdev(data_ddpg_s3)], capsize=yerr_width)
bar_hyb = ax.bar(hyb_index,
       [statistics.mean(data_hyb_s1), statistics.mean(data_hyb_s2), statistics.mean(data_hyb_s3)], bar_width, color=color_list[2],
       yerr=[statistics.stdev(data_hyb_s1), statistics.stdev(data_hyb_s2), statistics.stdev(data_hyb_s3)], capsize=yerr_width)

ax.plot([mpc_index[0]]*len(data_mpc_s1), data_mpc_s1, '.', color='black')
ax.plot([ddpg_index[0]]*len(data_ddpg_s1), data_ddpg_s1, '.', color='black')
ax.plot([hyb_index[0]]*len(data_hyb_s1), data_hyb_s1, '.', color='black')

ax.plot([mpc_index[1]]*len(data_mpc_s2), data_mpc_s2, '.', color='black')
ax.plot([ddpg_index[1]]*len(data_ddpg_s2), data_ddpg_s2, '.', color='black')
ax.plot([hyb_index[1]]*len(data_hyb_s2), data_hyb_s2, '.', color='black')

ax.plot([mpc_index[2]]*len(data_mpc_s3), data_mpc_s3, '.', color='black')
ax.plot([ddpg_index[2]]*len(data_ddpg_s3), data_ddpg_s3, '.', color='black')
ax.plot([hyb_index[2]]*len(data_hyb_s3), data_hyb_s3, '.', color='black')

ax.set_xticks(ddpg_index)
ax.set_xticklabels(['Scene 1', 'Scene 2', 'Scene 3'], fontsize=16)
ax.set_ylabel('Final step', fontsize=16)
ax.tick_params(axis='y', labelsize=16)

ax.legend((bar_mpc, bar_ddpg, bar_hyb), ('MPC', 'DDPG', 'Hybrid'), prop={'size': 16}, ncols=3)

ax.set_ylim(0, 220)
# plt.title('Final step comparison')
plt.show()