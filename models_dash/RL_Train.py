#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
#import torch.nn.functional as F
#import matplotlib.pyplot as plt
import pickle


# In[2]:


def base_cost(offer):
    return 0.8 * offer


# In[3]:


with open('./lr_cellphone_C.pkl','rb') as f:
#     # END
    model = pickle.load(f)


# In[4]:


data = pd.read_csv('./cellphones/cellphonedata.csv')
df = data[data.price_category == 'C'].copy().reset_index(drop=True)
df.drop(columns = ['ds','price_category'],inplace=True)
df['base_cost'] = df.offer.apply(lambda x: base_cost(x))

baseline_prices = df['olist_price'].values


# In[5]:


def demand_baseline(model, df_row, olist_price):
    
    year = df_row.year
    month = df_row.month
    dayofweek = df_row.dayofweek
    day = df_row.day
    olist_price = df_row.olist_price
    freight_value = df_row.freight_value
    competition_price = df_row.competition_price
    stock = df_row.stock
    black_friday = df_row.black_friday
    carnival = df_row.carnival
    christmas = df_row.christmas
    friday = df_row.friday
    mothers_day = df_row.mothers_day
    new_year = df_row.new_year
    others = df_row.others
    valentines = df_row.valentines

    X = np.array([year, month, dayofweek, day, olist_price, freight_value,
                 competition_price, stock, black_friday, carnival, christmas,
                 friday, mothers_day, new_year, others, valentines]).reshape(1,-1)
    
    #X = xgboost.DMatrix(X)
                 
    orders = model.predict(X)
    
    return max(orders[0],0)


# In[6]:


y_pred = []
for row in df.itertuples():
    y_pred.append(demand_baseline(model, row, 0)) 


# In[7]:


baseline = (df.olist_price + df.freight_value - df['base_cost']) * y_pred


# In[8]:


sum(baseline)


# In[9]:


sum(df['y'])
#sum(y_pred)


# In[10]:


data = pd.read_csv('./cellphones/cellphonedata.csv')
data = data[data.price_category == 'C'].copy().reset_index(drop=True) 
df = data[data.price_category == 'C'].copy().reset_index(drop=True)
df.drop(columns = ['ds','price_category'],inplace=True)
df['base_cost'] = df.offer.apply(lambda x: base_cost(x))

# cols = df.columns
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# normalized = scaler.fit_transform(df)

# df = pd.DataFrame(normalized, columns=cols)

df.drop(columns = ['offer'],inplace=True)

# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# X, y  = df.drop(columns=['base_cost', 'y']).values, df['y'].values
# model = model.fit(X, y)

df.drop(columns=['olist_price','y'], inplace=True)

# print(model.coef_)
# print(model.intercept_)

# df.tail()


# In[11]:


def predict_demand(model, df_row, olist_price):
    
    year = df_row.year
    month = df_row.month
    dayofweek = df_row.dayofweek
    day = df_row.day
    olist_price = olist_price
    freight_value = df_row.freight_value
    competition_price = df_row.competition_price
    stock = df_row.stock
    black_friday = df_row.black_friday
    carnival = df_row.carnival
    christmas = df_row.christmas
    friday = df_row.friday
    mothers_day = df_row.mothers_day
    new_year = df_row.new_year
    others = df_row.others
    valentines = df_row.valentines

    X = np.array([year, month, dayofweek, day, olist_price, freight_value,
                 competition_price, stock, black_friday, carnival, christmas,
                 friday, mothers_day, new_year, others, valentines]).reshape(1,-1)
    
    #X = xgboost.DMatrix(X)
                 
    orders = model.predict(X)
    
    return max(orders[0],0)


# In[12]:


class Environment:
    
    def __init__(self, model, df):
        
        self.model = model
        self.data = df
        self.N = len(self.data) - 1
        self.reset()
        
        
    def reset(self):
        self.t = 0
        self.done = False
        self.orders = 0
        self.olist_price = 0
        self.profits = 0
        return [self.olist_price, self.orders] + self.data.iloc[self.t].tolist() 
    
    def step(self, act):       
        
        # act = 0: stay, 1: raise, 2: lower
        if act == 0:
            self.olist_price = self.data['base_cost'][self.t] * 1.05
        elif act == 1:
            self.olist_price = self.data['base_cost'][self.t] * 1.075
        elif act == 2:
            self.olist_price = self.data['base_cost'][self.t] * 1.10
        elif act == 3:
            self.olist_price = self.data['base_cost'][self.t] * 1.125
        elif act == 4:
            self.olist_price = self.data['base_cost'][self.t] * 1.15
        elif act == 5:
            self.olist_price = self.data['base_cost'][self.t] * 1.175
        elif act == 6:
            self.olist_price = self.data['base_cost'][self.t] * 1.20
        elif act == 7:
            self.olist_price = self.data['base_cost'][self.t] * 1.225
        elif act == 8:
            self.olist_price = self.data['base_cost'][self.t] * 1.25
        elif act == 9:
            self.olist_price = self.data['base_cost'][self.t] * 1.275

        # Calculate demand
        self.orders = predict_demand(self.model, self.data.iloc[self.t], self.olist_price)        

        reward = (self.olist_price + self.data['freight_value'][self.t] - self.data['base_cost'][self.t])*self.orders
        self.profits += reward

        # set next time
        self.t += 1
        
        if (self.t == self.N):
            self.done=True

        return [self.olist_price, self.orders] + self.data.iloc[self.t].tolist(), reward, self.done # obs, reward, done 


# In[13]:


env = Environment(model,df)
env.reset()


# In[14]:


#def train_dqn(env):

#whats the return?
class Q_Network(nn.Module):
        
    def __init__(self,obs_len,hidden_size,actions_n):
            
        super(Q_Network,self).__init__()
            
#         self.fc_val = nn.Sequential(
#             nn.Linear(obs_len, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, actions_n)
#             # needs softmax?
#         )
        
        self.fc_val = nn.Sequential(
            nn.BatchNorm1d(num_features=obs_len),
            nn.Linear(obs_len, hidden_size),            
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(hidden_size, actions_n),
        )
        
        
    def forward(self,x):
        h =  self.fc_val(x)
        return (h)
            
            


# In[15]:


np.random.seed(36)
torch.manual_seed(36)

hidden_size = 50
input_size = 2 + df.shape[1]
output_size = 10 #5
LR = 0.001

epoch_num = 2
step_max = len(env.data) - 1
memory_size = 320 # 200
batch_size = 32
gamma = 0.9 # 0.97

epsilon = 1.0
epsilon_decrease = 1e-4
epsilon_min = 0.1
start_reduce_epsilon = 200
train_freq = 10
update_q_freq = 30 #20
show_log_freq = 5


# In[16]:


memory = []
total_step = 0
total_rewards = []
total_losses = []

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

Q = Q_Network(input_size, hidden_size, output_size).to(device=device)

Q_ast = copy.deepcopy(Q)

loss_function = nn.MSELoss()
optimizer = optim.Adam(list(Q.parameters()), lr=LR)

start = time.time()
for epoch in range(epoch_num):

    pobs = env.reset()
    step = 0
    done = False
    total_reward = 0
    total_loss = 0

    while not done and step < step_max:

        # select act

        pact = np.random.randint(10)
        if np.random.rand() > epsilon:
            #whats the return value?
            Q.eval()
            pact = Q(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)).to(device=device))
            pact = np.argmax(pact.data.cpu())
            pact = pact.numpy()
        
        # act
        obs, reward, done = env.step(pact)

        # add memory
        
        memory.append((pobs, pact, reward, obs, done))
        if len(memory) > memory_size:
            memory.pop(0)

        # train or update q
        if len(memory) == memory_size:
            if total_step % train_freq == 0:
                shuffled_memory = np.random.permutation(memory)
                memory_idx = range(len(shuffled_memory))
                for i in memory_idx[::batch_size]:
                    batch = np.array(shuffled_memory[i:i+batch_size])
                    b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                    b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                    b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                    b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                    b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)
                    
                    Q.train()
                    q = Q(torch.from_numpy(b_pobs).to(device=device))
                    q_ = Q_ast(torch.from_numpy(b_obs).to(device=device))
                    maxq = np.max(q_.data.cpu().numpy(), axis=1)
                    target = copy.deepcopy(q.data)
                    #import pdb; pdb.set_trace()

                    for j in range(batch_size):
                        target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                    Q.zero_grad()
                    loss = loss_function(q, target)
                    total_loss += loss.data.item()
                    loss.backward()
                    optimizer.step()
                    
            if total_step % update_q_freq == 0:
                Q_ast = copy.deepcopy(Q)
                
            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)

        #if (epoch+1) % show_log_freq == 0:
        if done or step == step_max:  
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            elapsed_time = time.time()-start
            print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
            start = time.time()
            
#return Q, total_losses, total_rewards


# In[17]:


torch.save(Q.state_dict(), './Q_state.torch')


# In[18]:


df.to_csv('./df.csv', index=False)

