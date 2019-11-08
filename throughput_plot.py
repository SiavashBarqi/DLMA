import numpy as np
import matplotlib.pyplot as plt

### calculate throughput
def cal_throughput(max_iter, N, reward):
	temp_sum = 0
	throughput = np.zeros(max_iter)
	for i in range(max_iter):
		if i < N:
			temp_sum     += reward[i] 
			throughput[i] = temp_sum / (i+1)
		else:
			temp_sum  += reward[i] - reward[i-N]
			throughput[i] = temp_sum / N
	return throughput

my_agent_throughputs = {}
agent_std = {}
agent_mean = {}
agent_rewards = []
aloha_rewards = {}
tdma_rewards = {}
agent_throughputs = []
aloha_throughputs = {}
tdma_throughputs = {}
sum_throughputs = {}
max_iter = 40000
N = 1000
numberOfDRLs = 3
x = np.linspace(0, max_iter, max_iter)
q = 0.2
for i in range(0, 4): # 0~3
	agent_rewards.append([])
	agent_throughputs.append([])
	for j in range(numberOfDRLs):
		agent_rewards[i].append(np.loadtxt('rewards/agent'+str(j)+'_M40_q0.4_t10-3_%d.txt' % (i+1)))
		agent_throughputs[i].append(cal_throughput(max_iter, N, agent_rewards[i][j]))
		
	#aloha_rewards[i]     = np.loadtxt('rewards/aloha_M40_q0.4_t10-3_%d.txt' % (i+1))
	#tdma_rewards[i]      = np.loadtxt('rewards/tdma_M40_q0.4_t10-3_%d.txt' % (i+1))
	#aloha_throughputs[i] = cal_throughput(max_iter, N, aloha_rewards[i])
	#tdma_throughputs[i]  = cal_throughput(max_iter, N, tdma_rewards[i])
	sum_throughputs[i]  = sum(agent_throughputs[i]) # + tdma_throughputs[i] + aloha_throughputs[i]

agent_optimal = np.ones(max_iter)*((1-q)*(1-0.3))
#aloha_optimal = np.ones(max_iter)*0
#tdma_optimal  = np.ones(max_iter)*(0.3*(1-q))

for j in range(numberOfDRLs):
	my_agent_throughputs[j] = np.array([agent_throughputs[0][j], agent_throughputs[1][j], agent_throughputs[2][j], agent_throughputs[3][j]])
#my_aloha_throughputs = np.array([aloha_throughputs[0], aloha_throughputs[1], aloha_throughputs[2], aloha_throughputs[3]])
#my_tdma_throughputs  = np.array([tdma_throughputs[0], tdma_throughputs[1], tdma_throughputs[2], tdma_throughputs[3]])
my_sum_throughputs  = np.array([sum_throughputs[0], sum_throughputs[1], sum_throughputs[2], sum_throughputs[3]])


for j in range(numberOfDRLs):
	agent_mean[j] = np.mean(my_agent_throughputs[j], axis=0)
	agent_std[j]  = np.std(my_agent_throughputs[j], axis=0)
#aloha_mean = np.mean(my_aloha_throughputs, axis=0)
#tdma_mean  = np.mean(my_tdma_throughputs, axis=0)
sum_mean  = np.mean(my_sum_throughputs, axis=0)
#aloha_std  = np.std(my_aloha_throughputs, axis=0)
#tdma_std   = np.std(my_tdma_throughputs, axis=0)
sum_std   = np.std(my_sum_throughputs, axis=0)


### plot
# fig = plt.figure(figsize=(10, 7))
fig = plt.figure()
ax  = fig.add_subplot(111)
sum_line, = plt.plot(sum_mean, color='y', lw=1, label='sum')
for j in range(numberOfDRLs):
	if j == 0: col = '4'
	elif j==1: col = 'e'
	else: col = 'c'
	agent_line, = plt.plot(agent_mean[j], color='#'+col+str(j*0)+str(j*3)+str(j*3)+str(j*3)+str(j*3), lw=1, label='agent')
"""agent_optimal_line, = plt.plot(agent_optimal, color='r', lw=3, label='agent optimal')
aloha_line, = plt.plot(aloha_mean, color='b', lw=1, label='aloha')
#aloha_optimal_line, = plt.plot(aloha_optimal, color='b', lw=3, label='aloha optimal')
tdma_lin3,  = plt.plot(tdma_mean,  color='g', lw=1, label='tdma')
#tdma_optimal_line,  = plt.plot(tdma_optimal, color='g', lw=3, label='tdma  optimal')
"""
handles, labels = ax.get_legend_handles_labels()
# plt.legend(handles=[agent_line, aloha_line, tdma_lin3, agent_optimal_line, aloha_optimal_line, tdma_optimal_line], loc='best')
plt.legend(handles, labels, loc='upper right', ncol=2, bbox_to_anchor=(0.5,1), fancybox=True, shadow=True)

for j in range(numberOfDRLs):
	if j == 0: col = '4'
	elif j==1: col = 'e'
	else: col = 'c'
	plt.fill_between(x, agent_mean[j]-agent_std[j], agent_mean[j]+agent_std[j],    
		alpha=0.4, edgecolor='#'+col+str(j*0)+str(j*3)+str(j*3)+str(j*3)+str(j*3), facecolor='#'+col+str(j*0)+str(j*3)+str(j*3)+str(j*3)+str(j*3),
		linewidth=4, linestyle='dashdot', antialiased=True)
"""plt.fill_between(x, aloha_mean-aloha_std, aloha_mean+aloha_std,    
    alpha=0.4, edgecolor='#0343df', facecolor='#0343df',
    linewidth=4, linestyle='dashdot', antialiased=True)
plt.fill_between(x, tdma_mean-tdma_std, tdma_mean+tdma_std,    
    alpha=0.4, edgecolor='#15b01a', facecolor='#15b01a',
    linewidth=4, linestyle='dashdot', antialiased=True)"""
plt.fill_between(x, sum_mean-sum_std, sum_mean+sum_std,    
    alpha=0.4, edgecolor='#51af10', facecolor='#51af10',
    linewidth=4, linestyle='dashdot', antialiased=True)

plt.grid()
plt.xlabel('Time steps')
plt.ylabel('Throughput')
plt.xlim((0, max_iter))
plt.ylim((0, 1))
plt.show()


