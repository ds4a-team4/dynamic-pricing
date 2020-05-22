import torch
import torch.nn as nn

class Q_Network(nn.Module):
        
    def __init__(self,obs_len,hidden_size,actions_n):
            
        super(Q_Network,self).__init__()
            
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
            
            