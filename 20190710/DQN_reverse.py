import random
from reverse_parking import World
from reverse_parking import Window
import sys,os
import pygame
import numpy as np
import random
import math
from pygame.locals import *
from pygame.color import THECOLORS

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms 
import time
import matplotlib.pyplot as plt 
from plot import *
import argparse



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
    def __init__(self, n_states, n_actions, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity, resize_number, iteration, target_net, eval_net):
        self.memory = []
        
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
        self.eval_net, self.target_net = eval_net, target_net
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr= lr, eps= 1e-8)
        self.loss_func = nn.MSELoss()
        
    def choose_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            action = np.random.randint(0,self.n_actions)

        else:
            #state = torch.FloatTensor(state)
            #x = torch.cat((state,state,state,state)).view(4,84,84).unsqueeze(0)
            #x = torch.unsqueeze(torch.cuda.FloatTensor(state), 0)
            #print(x.shape)
            if torch.cuda.is_available():
                x = torch.cuda.FloatTensor(state).unsqueeze(0)
                action_value = self.eval_net(x)
                action = torch.max(action_value, 1)[1].data.cpu().numpy()[0] 
            else:
                #print("cpu")
                x = torch.FloatTensor(state).unsqueeze(0)
                action_value = self.eval_net(x)
                action = torch.max(action_value, 1)[1].data.numpy()[0] 
                
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
        b_state = np.concatenate(tuple(d[0][np.newaxis, :] for d in b_memory ))
        #b_state = torch.from_numpy(b_state)
        b_state = torch.FloatTensor(b_state)

        b_action = np.zeros(len(b_memory))
        for i in range(len(b_memory)):
            b_action[i] = b_memory[i][1]
        b_action = torch.LongTensor(b_action).view(-1,1)
        
        b_reward = np.zeros(len(b_memory))
        for i in range(len(b_memory)):
            b_reward[i] = b_memory[i][2]
        b_reward = torch.FloatTensor(b_reward).view(-1,1)

        #b_reward = torch.cat(tuple(d[2] for d in b_memory ))
        b_next_state = np.concatenate(tuple(d[3][np.newaxis, :] for d in b_memory ))
        b_next_state = torch.FloatTensor(b_next_state)




        if torch.cuda.is_available():  # put on GPU if CUDA is available
            b_state = b_state.cuda()
            b_action = b_action.cuda()
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


def get_epsilon(iteration, episode, epsilon_rate, min_epsilon, memory_capacity):
    if iteration < memory_capacity:
        return 1
    else:
        return max(min_epsilon, round(1 - episode*epsilon_rate,9))
    #return max(0.1, round(1 - episode*epsilon_rate,7))

def image_to_tensor(image):
    resize_number = 84
    image_data = cv2.cvtColor(cv2.resize(image, (resize_number, resize_number)), cv2.COLOR_BGR2GRAY)
    image_data = np.reshape(image_data, (1,resize_number, resize_number))
    image_data = image_data.astype(np.float32)
    #image_tensor = torch.from_numpy(image_data)
    #if torch.cuda.is_available():  # put on GPU if CUDA is available
     #   image_tensor = image_tensor.cuda()
    return image_data   


def plot2D(x,y,x_label,y_label,now_iteration):
    fig = plt.figure()
    plt.plot(x,y)                                                                                                                   
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.show()
    fig.savefig(x_label+"_"+y_label+"_"+now_iteration+".png")

def save_csv(path, name, data):
    filename = name+".csv"
    path = path+filename
    #print(path)
    text = open(path, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id",str(name)])
    for i in range(len(data)):
        s.writerow([i,data[i]]) 
    text.close()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def show_text(text, x, y):#專門顯示文字的方法，除了顯示文字還能指定顯示的位置
    #font = pygame.font.Font('Fonts/notosansthaana', 24)#本文主角
    x = x
    y = y
    #text = font.render(text, True, (255, 255, 255))
    screen.blit(text, (x, y))
    pygame.display.update()
    
    

def training(dqn, env, network_env, min_epsilon):
    n_episodes = 50000000 
    average_rewards = 0    
    Q_number_list= []
    Q_number = 0
    episode_list = []
    Q_max_list = []
    reward_list = []
    iteration=0
    
    pygame.init()
    window_size = (640, 480)
    window=Window(window_size)
    screen=window.surface
    window.update_title(__file__)
    myclock=pygame.time.Clock()       
    framerate_limit=20  
    t=0.0
    window.surface.fill(THECOLORS["black"])
    dt=float(myclock.tick(framerate_limit) * 1e-3)
    

    for i_episode in range(n_episodes):
        positiv_enumber = 0
        
        new_epsilon = get_epsilon( dqn.iteration,i_episode, epsilon_rate, min_epsilon, dqn.memory_capacity)
        #new_epsilon = 1
        
        action = 1
#        image_data = env.reset()
#        image_data = image_to_tensor(image_data)
        state = np.zeros((4,84,84))
#        state[3], state[2], state[1], state[0] = state[2], state[1], state[0], image_data
        next_state = np.zeros((4,84,84))
        rewards = 0
        if new_epsilon == min_epsilon:
            print(i_episode)
            break
        
        while True:
            #env.render()   
            
            action = dqn.choose_action(state, new_epsilon)
            
            image_data, reward, done = env.step(action)
            image_data = image_to_tensor(image_data)
            next_state[3], next_state[2], next_state[1], next_state[0] = state[2], state[1], state[0], image_data
            dqn.store_transition(state, action, reward, next_state)
            rewards += reward
            state = next_state 
            dqn.iteration += 1
            #pygame.time.delay(100)
            env.draw(screen)
            
            
            if reward >0:
                positiv_enumber +=1
 
            if dqn.iteration > dqn.memory_capacity and dqn.iteration % learning_frequency ==0  or new_epsilon < min_epsilon+epsilon_rate:
                Q_max = dqn.learn()
                Q_max_list.append(Q_max)
                Q_number_list.append(Q_number)
                Q_number +=1
                
                
            
            if dqn.iteration % 20000 ==0 or new_epsilon < min_epsilon+epsilon_rate:                            
                torch.save(dqn.target_net.state_dict(), model_path + "/target_model" +str(dqn.iteration) +".pkl")
                torch.save(dqn.eval_net.state_dict(), model_path + "/eval_model" +str(dqn.iteration)+ ".pkl")
                
                save_csv(csv_path, "/average_reward", reward_list)
                save_csv(csv_path, "/Q_max", Q_max_list)
                
                
                
                plot2D(episode_list,reward_list, graph_path + "/training_episode", "rewards", str(dqn.iteration))
                plot2D(Q_number_list,Q_max_list, graph_path + "/training_episode", "Q_max", str(dqn.iteration))
                #torch.save(dqn, "current_model_" + str(dqn.iteration) + ".pth")
                print("save model at "+str(dqn.iteration)+" iteration")
            
            if done:
                end = time.time()
                now = round(end - start,2)
                
                env.reset()
                env.draw(screen)
                if i_episode % 100 ==0 and i_episode!=0:
                    average_rewards += rewards
                    print("episode: ", i_episode,"new_epsilon",new_epsilon,str(i_episode-100)+"~"+str(i_episode-1)+"_average_reward", average_rewards/100)
                    print('Episode finished after {} timesteps, total rewards {}, positive number {}'.format(now, rewards, positiv_enumber))
                    reward_list.append(average_rewards/100)
                    episode_list.append(i_episode)
                    average_rewards = 0
                else:
                    print("episode: ", i_episode,", iteration: ",dqn.iteration, ", new_epsilon",new_epsilon)
                    print('Episode finished after {} timesteps, total rewards {}, positive number {}'.format(now, rewards, positiv_enumber))
                    average_rewards += rewards
                

                    
                #print(len(reward_list),episode_list)
                break            
            #pygame.display.flip()



def testing(dqn, env):
    n_episodes = 2000000  
    fps = 20
    pygame.init()
    window_size = (640, 480)
    window=Window(window_size)
    screen=window.surface
    window.update_title(__file__)
    myclock=pygame.time.Clock()       
    framerate_limit=20  
    t=0.0
    window.surface.fill(THECOLORS["black"])
    dt=float(myclock.tick(framerate_limit) * 1e-3)

    for i_episode in range(n_episodes):
        random_action_counter = 0    
        #image_data = env.reset()
        #image_data = image_to_tensor(image_data)
        state = np.zeros((4,84,84))
        #state[3], state[2], state[1], state[0] = state[2], state[1], state[0], image_data
        next_state = np.zeros((4,84,84))
        
        rewards = 0
        while True:
            """
            if random_action_counter >= 80:
                env.render()

            if random_action_counter < 100 :
                action = dqn.choose_action(state, 1)
                random_action_counter += 1
            else:
                action = dqn.choose_action(state, 0)
            """
            action = dqn.choose_action(state, 0)
            image_data, reward, done = env.step(action)
            image_data = image_to_tensor(image_data)
            next_state[3], next_state[2], next_state[1], next_state[0] = state[2], state[1], state[0], image_data            
            state = next_state 
            time.sleep(3.0/fps)
            env.draw(screen)
            rewards += reward
            font = pygame.font.SysFont('SimHei',40)
            surface1 = font.render("total rewards" , True, [255, 0, 0]) 
            surface2 = font.render(str(rewards) , True, [255, 0, 0])
            screen.blit(surface1, [450, 0])
            screen.blit(surface2, [520, 30])
            if done:
                env.reset()
                env.draw(screen)
                break
            pygame.display.flip()
   # env.close()






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='N',
                        help='gamma (default: 0.99)')
    parser.add_argument('--target_replace_iter', type=int, default=5000, metavar='LR',
                        help='target_replace_iter (default: 10000)')
    parser.add_argument('--memory_capacity', type=int, default=200000, metavar='M',
                        help='memory_capacity (default: 300000)')
    parser.add_argument('--resize_number', type=int, default=84, metavar='S',
                        help='image_resize_number (default: 84)')
    parser.add_argument('--learning_frequency', type=int, default=4, metavar='N',
                        help='learning frequency')
    parser.add_argument('--epsilon_rate', type=float, default=0.000001, metavar='N',
                        help='epsilon_rate')
    parser.add_argument('--store_path', type=str, default="new_train/", metavar='N',
                        help='store_path')
    parser.add_argument('--mode', type=str, default="training", metavar='N',
                        help='mode')
    parser.add_argument('--target_path', type=str, default="", metavar='N',
                        help='mode')
    parser.add_argument('--eval_path', type=str, default="", metavar='N',
                        help='mode')
    parser.add_argument('--min_epsilon', type=float, default=0.05, metavar='N',
                        help='mode')
    args = parser.parse_args()

    n_states = 84
    batch_size = args.batch_size
    lr = args.lr                 
    initial_epsilon = 1             
    gamma = args.gamma               
    target_replace_iter = args.target_replace_iter
    memory_capacity = args.memory_capacity  
    resize_number = args.resize_number
    learning_frequency = args.learning_frequency
    epsilon_rate = args.epsilon_rate
    store_path = args.store_path
    mode = args.mode
    min_epsilon = args.min_epsilon
    iteration = 0
    
        
   
    start = time.time()        


    env = World()   
    network_env = World()
    #env = gym.make('Breakout-v0')
    #network_env = gym.make('Breakout-v0')
    n_actions = 11      
    delta = env.delta
    limit_length = env.limit_length
    limit_width  = env.limit_width       
    close_right = env.close_right
    close_left = env.close_left
    close_under = env.close_under
    close_above = env.close_above
    correct_right = env.correct_right
    correct_left = env.correct_left
    correct_under = env.correct_under
    correct_above = env.correct_above
    correct_angle = env.correct_angle
        
    model = Net(n_actions)  
    if torch.cuda.is_available():
        model = model.cuda()
    eval_net, target_net = model, model    
 
#mkdir folder and creat csv file   
    model_path = os.path.join("model", store_path)
    graph_path = os.path.join("graph", store_path)
    csv_path = os.path.join("csv", store_path)
    print(model_path)
    print(graph_path)
    print(csv_path)
    print("............")                               
    mkdir(graph_path)
    mkdir(model_path)
    mkdir(csv_path)          
    filename = "csv/" + store_path+ "/parameters.csv"
    text = open(filename, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["mode", "memory_capacity", "epsilon_rate", "learning_frequency", "target_replace_iter", "resize_number", "min_epsilon", "batch_size", "delta", "limit_length", "limit_width",  "gamma", "lr", "close_right", "close_left", "close_under", "close_above", "correct_right", "correct_left", "correct_under", "correct_above", "correct_angle"])
    s.writerow([mode, memory_capacity, epsilon_rate, learning_frequency, target_replace_iter, resize_number, min_epsilon, batch_size, delta, limit_length, limit_width, gamma, lr, close_right, close_left, close_under, close_above, correct_right, correct_left, correct_under, correct_above, correct_angle])
    text.close()

        


    if mode == "training":
        
        dqn = DQN(n_states, n_actions, batch_size, lr, initial_epsilon, gamma, target_replace_iter, memory_capacity, resize_number, iteration, target_net, eval_net)

        training(dqn, env, network_env, min_epsilon)

    elif mode == "testing":
        eval_path = args.eval_path
        target_path = args.target_path
        eval_net.load_state_dict(torch.load(eval_path))
        target_net.load_state_dict(torch.load(target_path))
        eval_net.eval()
        target_net.eval()
        dqn = DQN(n_states, n_actions, batch_size, lr, initial_epsilon, gamma, target_replace_iter, memory_capacity, resize_number, iteration, target_net, eval_net)

        testing(dqn, env)


"""
env = World()
pygame.init()
window_size = (640, 480)
window=Window(window_size)
screen=window.surface
window.update_title(__file__)
myclock=pygame.time.Clock()       
framerate_limit=20  
t=0.0
window.surface.fill(THECOLORS["black"])
dt=float(myclock.tick(framerate_limit) * 1e-3)


for i in range(100):
    action = random.randint(0, 10)
    image, reward, terminal = env.step(action)
    print(action)
    print(reward, terminal)
    pygame.time.delay(500)
    env.draw(screen)
    if terminal:
        env.reset()
        env.draw(screen)
    pygame.display.flip()
"""