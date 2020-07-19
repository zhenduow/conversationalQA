import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from transformers import *
import warnings 
warnings.filterwarnings("ignore")

class LinearDeepQNetwork(nn.Module):
    '''
    The linear deep Q network used by the agent.
    '''
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions


class Agent():
    '''
    The conversational QA agent.
    '''
    def __init__(self, input_dims, n_actions, lr, gamma=0.99,
                 epsilon=1.0, eps_dec=1e-3, eps_min=0.01):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.embedding_model = XLNetModel.from_pretrained('xlnet-base-cased')

        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        
        # Encode text
        input_ids = T.tensor([self.tokenizer.encode(observation, add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        with T.no_grad():
            encoded_observation = self.embedding_model(input_ids)[0][0][0]

        if np.random.random() > self.epsilon:
            'if the random number is greater than exploration threshold, choose the action maximizing Q'
            state = T.tensor(encoded_observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            #action = T.argmax(actions).item()
            action = actions.argmax()
        else:
            'randomly choosing an action'
            action = np.random.choice(self.action_space)

        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        
        # Encode text
        input_ids = T.tensor([self.tokenizer.encode(state, add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        input_ids_ = T.tensor([self.tokenizer.encode(state_, add_special_tokens=True)])
        with T.no_grad():
            encoded_state = self.embedding_model(input_ids)[0][0][0]  
            encoded_state_ = self.embedding_model(input_ids_)[0][0][0]  

        self.Q.optimizer.zero_grad()
        states = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(encoded_state_, dtype=T.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[actions]

        q_next = self.Q.forward(states_).max()

        q_target = reward + self.gamma*q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()
