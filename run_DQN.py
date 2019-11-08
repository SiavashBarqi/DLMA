from environment import ENVIRONMENT
from DQN_brain import DQN

import numpy as np
import matplotlib.pyplot as plt
import time

NN = 3

def main(max_iter):
    print('------------------------------------------')
    print('---------- Start processing ... ----------')
    print('------------------------------------------')

    state = {}

    for k in range(4):
        numberOfDRLs = 1
        start = time.time()
        agent_reward_list = []
        aloha_reward_list  = []
        tdma_reward_list  = []
        agent_action = [0 for i in range(NN)]
        for j in range(NN):
            state[j] = env.reset()
        state_length = len(state[0])
        env.tdma(k)
        for i in range(max_iter):
            if i >= 0:#20000:
                if i >= 0:#40000:
                    numberOfDRLs = NN
                elif i > 20000:
                    numberOfDRLs = NN-1
                for j in range(numberOfDRLs):
                    agent_action[j] = dqn_agent.choose_action(state[j])
            else:
                agent_action = [0 for i in range(NN)]
            observation_, reward, agent_reward, aloha_reward, tdma_reward = env.step(agent_action)
            agent_reward_list.append(agent_reward)
            #aloha_reward_list.append(aloha_reward)
            #tdma_reward_list.append(tdma_reward)
            if i < 2000000:    
                for j in range(numberOfDRLs):
                    if state_length<3:
                        next_state = np.concatenate([agent_action[j], observation_[j]])
                    else:
                        next_state = np.concatenate([state[j][2:], [agent_action[j], observation_[j]]])

                    dqn_agent.store_transition(state[j], agent_action[j], reward, next_state) # SO IMPORTANT!!!!

                    if i > 200: # 20000+200:
                        dqn_agent.learn()       # internally iterates default (prediction) model
                    state[j] = next_state

        for j in range(numberOfDRLs):
            with open('rewards/agent' + str(j) + '_M40_q0.4_t10-3_%d.txt' %(k+1), 'w') as my_agent:
                for i in agent_reward_list:
                    my_agent.write(str(i[j]) + '   ')
        """with open('rewards/aloha_M40_q0.4_t10-3_%d.txt' %(k+1), 'w') as my_aloha:
            for i in aloha_reward_list:
                my_aloha.write(str(i) + '   ') 
        with open('rewards/tdma_M40_q0.4_t10-3_%d.txt' %(k+1), 'w') as my_tdma:
            for i in tdma_reward_list:
                my_tdma.write(str(i) + '   ')"""
        # save model 
        # dqn_agent.save_model("models/model_len2e5_M20_h6_q0.6_3.h5")  
        # print the results
        print('-----------------------------')
        for j in range(numberOfDRLs):
            print('average agent'+str(j)+' reward: {}'.format(np.mean(agent_reward_list[-2000:][j])))
        #print('average aloha reward: {}'.format(np.mean(aloha_reward_list[-2000:])))
        #print('average tdma  reward: {}'.format(np.mean(tdma_reward_list[-2000:])))
        print('average total reward: {}'.format(np.mean(agent_reward_list[-2000:])))# + 
                                                #np.mean(aloha_reward_list[-2000:]) +
                                                #np.mean(tdma_reward_list[-2000:])))
        #print('tdma prob: %i' % env.tdmaPrb)
        #print('aloha prob: %i' % env.alohaPrb)
        print('agent prob: %i' %env.agentPrb)
        print('Time elapsed:', time.time()-start)



if __name__ == "__main__":
    env = ENVIRONMENT(state_size=NN*40, 
                      attempt_prob=0.2,
                      )
                      
    dqn_agent = DQN(env.state_size,
                    env.n_actions,  
                    env.n_nodes,
                    memory_size=NN*500,
                    replace_target_iter=200,
                    batch_size=NN*32,
                    learning_rate=0.01,
                    gamma=0.9,
                    epsilon=0.1,
                    epsilon_min=0.005,
                    epsilon_decay=0.995,
                    )

    main(max_iter=40000)