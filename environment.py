import numpy as np 
from numpy import *

Tx_time = 1000000
idleness_time = 0
counter = 0
check = 0
class ENVIRONMENT(object):
	"""docstring for ENVIRONMENT"""
	def __init__(self,
				 state_size = 10,
				 attempt_prob = 1,
				 ):
		super(ENVIRONMENT, self).__init__()
		self.state_size = state_size
		self.attempt_prob = attempt_prob # aloha node attempt probability
		self.action_space = ['w', 't'] # w: wait t: transmit
		self.n_actions = len(self.action_space)
		self.n_nodes = 3
		self.tdmaPrb = 0
		self.alohaPrb = 0
		self.agentPrb = 0
		self.action_list = [0, 0, 0, 0, 0]#, 0, 0, 0, 0, 0] # case0

		# self.action_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] # case1.1
		# self.action_list = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # case1.2
		# self.action_list = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # case1.3
		
		# self.action_list = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0] # case2.1 
		# self.action_list = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0] # case2.2 
		# self.action_list = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1] # case2.3
	
		
		# self.action_list = [0, 1, 1, 0, 0, 0, 0, 0, 1, 1] # case4.2 
		# self.action_list = [0, 0, 1, 1, 1, 0, 0, 1, 0, 0] # case4.3


		# self.action_list = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # case5.1 
		# self.action_list = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1] # case5.2 
		# self.action_list = [0, 1, 1, 1, 0, 1, 0, 1, 0, 0] # case5.3

		# self.action_list = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0] # case6.1
		# self.action_list = [1, 1, 0, 1, 1, 0, 1, 1, 0, 0] # case6.2
		# self.action_list = [0, 1, 1, 1, 0, 1, 0, 1, 0, 1] # case6.3
	
		# self.action_list = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0] # case7.1 
		# self.action_list = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1] # case7.2
		# self.action_list = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1] # case7.3

		# self.action_list = [1, 0, 1, 1, 1, 1, 1, 1, 0, 1] # case8.1 
		# self.action_list = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] # case8.2
		# self.action_list = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1] # case8.3

		# self.action_list = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1] # case9.1
		# self.action_list = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1] # case9.2
		# self.action_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0] # case9.3

		# self.action_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # case10

	def reset(self):
		init_state = np.zeros(self.state_size, int)
		q = self.attempt_prob
		self.tdmaPrb = 2*q*(1-q)*.2 + .4*(1-q)**2
		self.alohaPrb = .2*q*( (1-q)**2 )*3 + .4*(q**2)*(1-q)*3 + .6*(pow(q,3))
		#self.agentPrb = .2*(1-q)*3 + .4*((1-q)**2)*3 + .6*pow((1-q),3)
		self.agentPrb = (.6*.8 + 1.2*.64 + .6*.64*.8)/(1.2)
		return init_state

	def tdma(self, k):
		listt = [0, 0, 0, 0]
		listt[0] = [0, 1, 0, 1, 0]#, 0, 1, 0, 0, 0] # case3.1 
		listt[1] = [1, 0, 1, 0, 0]#, 1, 0, 0, 0, 0] # case3.2 
		listt[2] = [0, 1, 0, 0, 1]#, 0, 0, 0, 0, 0] # case3.3
	
		listt[3] = [1, 0, 0, 0, 1]#, 0, 1, 0, 0, 0] # case4.1 
		self.action_list = listt[k]

	def step(self, action):
		global counter

		reward = 0
		n = len(action)
		agent_reward = [0 for j in range(n)]
		aloha_reward = 0
		tdma_reward = 0
		observation_ = [0 for j in range(n)]
		#cap = self.cap
		cap = 1
		
		if np.random.random()<self.attempt_prob:
			aloha_action = 1
		else:
			aloha_action = 0
		tdma_action = self.action_list[counter]
		"""
		cap -= aloha_action
		cap -= tdma_action
		"""
		if cap == 1:
			idleness_time += 1
		else:
			idleness_time = check

		cap -= sum(action)

		for j in range(n):
			if action[j] == 1:
				if cap == 0:
					# print('agent success')
					reward = 1
					agent_reward[j] = 1
					observation_[j] = 1 # tx, success
				else:
					# print('collision')
					# reward = 0
					observation_[j] = -1 # tx, no success
			else:
				if cap == 0:
					reward = 1
					observation_[j] = 2 # no tx, success
					### Just for figure
					aloha_reward = aloha_action
					tdma_reward = tdma_action
				elif cap == 1:
					# idle: no tx, no success
					observation_[j] = 0
				else:
					# print('aloha-tdma collision')
					observation_[j] = -2 # no tx, no success

		counter += 1
		if counter == len(self.action_list): 
			counter = 0
		#self.cap = cap
		return observation_, reward, agent_reward, aloha_reward, tdma_reward




