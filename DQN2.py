import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms 
import random
#from reverse_parking_env import World
import gym
import time
import matplotlib.pyplot as plt 
from plot import *

#CNN
class Net(nn.Module):
    def __init__(self, n_actions):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, n_actions)        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
    
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv3(x)).view(x.size()[0], -1)
              
        x = F.relu(self.fc4(x))
        
        x = F.relu(self.fc5(x))
       
        return x

#DQN
class DQN(object):
    def __init__(self, n_states, n_actions, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity, resize_number, iteration):
        self.eval_net, self.target_net = Net(n_actions).cuda(), Net(n_actions).cuda()         
        self.memory = []
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.iteration = 0 
        self.resize_number = resize_number
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter        
        self.memory_capacity = memory_capacity
        
    def choose_action(self, state,epsilon):
        if np.random.uniform() < epsilon:
            action = np.random.randint(0,self.n_actions)
        else:
            #state = torch.FloatTensor(state)
            #x = torch.cat((state,state,state,state)).view(4,84,84).unsqueeze(0)
            #x = torch.unsqueeze(torch.cuda.FloatTensor(state), 0)
            #print(x.shape)
            x = torch.cuda.FloatTensor(state)
            action_value = self.eval_net(x)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()[0] 

        return action
        
        
    def store_transition(self, state, action, reward, next_state):
     
        #action = torch.cuda.IntTensor(action)
        
        #reward = torch.cuda.IntTensor(int(reward+1))
        if self.memory_counter < self.memory_capacity:
            #print(self.memory_counter)
            self.memory.append((state,action,reward,next_state))
            self.memory_counter += 1 
            
        else:
            #print("full")
            self.memory.pop(0)
            self.memory.append((state,action,reward,next_state))
            self.memory_counter += 1



    def learn(self):
        b_memory = random.sample(self.memory, min(len(self.memory),batch_size))
        #for i in range(32):
         #   print(b_memory[i][0].shape,"xxxxxx",b_memory[i][1],"xxxxxxxxx",b_memory[i][2],"xxxxxxxxxxx",b_memory[i][3].shape,"xxxxxxxxxxxx")
        #print(len(b_memory))
        b_state = torch.cat(tuple(d[0] for d in b_memory ))
        b_action = np.zeros(len(b_memory))
        for i in range(len(b_memory)):
            b_action[i] = b_memory[i][1]
        b_action = torch.LongTensor(b_action)
        
        b_reward = np.zeros(len(b_memory))
        for i in range(len(b_memory)):
            b_reward[i] = b_memory[i][2]
        b_reward = torch.FloatTensor(b_reward)
        #b_reward = torch.cat(tuple(d[2] for d in b_memory ))
        b_next_state = torch.cat(tuple(d[3] for d in b_memory ))



        if torch.cuda.is_available():  # put on GPU if CUDA is available
            b_state = b_state.cuda()
            b_action = b_action.cuda().view(-1,1)
            b_reward = b_reward.cuda()
            b_next_state = b_next_state.cuda()
        
        #Compute loss between Q values of eval net & target net
        q_eval = self.eval_net(b_state).gather(1,b_action) # evaluate the Q values of the experiences, given the states & actions taken at that time
                
        q_next = self.target_net(b_next_state).detach() # detach from graph, don't backpropagate
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # compute the target Q values
        loss = self.loss_func(q_eval, q_target)
        #print("---------------------------")
        #print(q_next.max(1)[0].view(self.batch_size, 1).cpu().numpy())
        #print("----------------------------------------------------------")
        #print(q_next.max(1)[0].view(self.batch_size, 1).cpu().numpy().max())
        #print(q_next.max(1)[0].view(self.batch_size, 1).cpu().numpy().mean())
        #print(q_next.max(1)[0].view(self.batch_size, 1).cpu().numpy().max()-q_next.max(1)[0].view(self.batch_size, 1).cpu().numpy().mean())
        return round(q_next.max(1)[0].view(self.batch_size, 1).cpu().numpy().max(),4)
        
        """
        output_batch = (b_next_state)
        print(len(output_batch))
        #print(len(b_reward),len(b_memory), len(output_batch))
        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(b_reward[i] if b_memory[i][4]
                                  else b_reward[i] + self.gamma * torch.max(output_batch[i])
                                  for i in range(len(b_memory))))

        # extract Q-value
        q_value = torch.sum(Net(b_state) * b_action, dim=1)
        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        """

        # Backpropagation
        dqn.optimizer.zero_grad()
        loss.backward()
        dqn.optimizer.step()

        # target parameter update
        if self.iteration % self.target_replace_iter == 0:
            dqn.target_net.load_state_dict(dqn.eval_net.state_dict())


def get_epsilon(episode):
    return max(0.1, round(1 - episode*0.000014,7))

def image_to_tensor(image):
    resize_number = 84
    image_data = cv2.cvtColor(cv2.resize(image, (resize_number, resize_number)), cv2.COLOR_BGR2GRAY)
    image_data = np.reshape(image_data, (1,resize_number, resize_number))
    image_data = image_data.astype(np.float32)
    image_tensor = torch.from_numpy(image_data)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor   


def plot2D(x,y,x_label,y_label,now_iteration):
    fig = plt.figure()
    plt.plot(x,y)                                                                                                                   
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.show()
    fig.savefig(x_label+"_"+y_label+"_"+now_iteration+".jpg")

def save_csv(path, name, data):
    filename = name+".csv"
    text = open(os.path.join(path,filename), "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id",str(name)])
    for i in range(len(data)):
        s.writerow([i,data[i]]) 
    text.close()


if __name__ == '__main__':
    start = time.time()    
    
    env = gym.make('MsPacman-v0')
    n_actions = env.action_space.n      
    
    
    Q_number_list= []
    Q_number = 0
    episode_list = []
    Q_max_list = []
    reward_list = []
    
    action_space = ["1","2","3","4","5","a","s","d","f","g","none"]
    #n_actions = len(action_space)
    n_states = 84
    batch_size = 32
    lr = 0.01                 
    initial_epsilon = 1             
    gamma = 0.99               
    target_replace_iter = 10000
    memory_capacity = 60000
    n_episodes = 2000000   
    resize_number = 84
    iteration=0
    learning_frequency = 4
    dqn = DQN(n_states, n_actions, batch_size, lr, initial_epsilon, gamma, target_replace_iter, memory_capacity, resize_number, iteration)
    average_rewards = 0    
    
    for i_episode in range(n_episodes):
        

        rewards = 0
        new_epsilon = get_epsilon(i_episode)
        action = 1
        #image_data, reward, done = env.step(action)
        image_data = env.reset()
        image_data = image_to_tensor(image_data)
        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)      
        while True:
            env.render()   
            
            action = dqn.choose_action(state, new_epsilon)
            
            image_data, reward, done, information = env.step(action)
            image_data = image_to_tensor(image_data)
            next_state = torch.cat((image_data, image_data, image_data, image_data),0).unsqueeze(0) 
            dqn.store_transition(state, action, reward, next_state)
            rewards += reward
            
            state = next_state 
            dqn.iteration += 1
            
            if dqn.iteration > memory_capacity and dqn.iteration % learning_frequency ==0:
                Q_max = dqn.learn()
                Q_max_list.append(Q_max)
                Q_number_list.append(Q_number)
                Q_number +=1

                
            
            if dqn.iteration % 5000000 ==0:
                torch.save(dqn.target_net.state_dict(), "model/target_model_" + str(dqn.iteration) +".pkl")
                torch.save(dqn.eval_net.state_dict(), "model/eval_model_" + str(dqn.iteration) +".pkl")
                
                save_csv("graph", "average_reward", reward_list)
                save_csv("graph", "Q_max", Q_max_list)
                plot2D(episode_list,reward_list,"graph/training_episode","rewards",str(dqn.iteration))
                plot2D(Q_number_list,Q_max_list,"graph/training_episode","Q_max",str(dqn.iteration))
                #torch.save(dqn, "current_model_" + str(dqn.iteration) + ".pth")
                print("save model at "+str(dqn.iteration)+" iteration")
            
            if done:
                end = time.time()
                now = round(end - start,2)

                if i_episode % 100 ==0 and i_episode!=0:
                    average_rewards += rewards
                    print("episode: ", i_episode,"new_epsilon",new_epsilon,str(i_episode-100)+"~"+str(i_episode-1)+"_average_reward", average_rewards/100)
                    print('Episode finished after {} timesteps, total rewards {}'.format(now, rewards))
                    reward_list.append(average_rewards/100)
                    episode_list.append(i_episode)
                    average_rewards = 0
                else:
                    print("episode: ", i_episode,", iteration: ",dqn.iteration, ", new_epsilon",new_epsilon)
                    print('Episode finished after {} timesteps, total rewards {}'.format(now, rewards))
                    average_rewards += rewards
                    
                #print(len(reward_list),episode_list)
                break
            #print(dqn.iteration,dqn.epsilon)
        
    env.close()
