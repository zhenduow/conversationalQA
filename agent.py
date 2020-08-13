import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from transformers import *
import warnings 
import random
warnings.filterwarnings("ignore")

class LinearDeepQNetwork(nn.Module):
    '''
    The linear deep Q network used by the agent.
    '''
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        score = self.fc2(hidden)

        return score


class Agent():
    '''
    The conversational QA agent.
    '''
    def __init__(self, input_dims, n_actions, lr, gamma=0.25,
                 epsilon=1.0, eps_dec=1e-3, eps_min=0.01, top_k = 1, data_augment = 10):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.top_k = top_k
        self.data_augment = data_augment
        self.action_space = [i for i in range(self.n_actions)]
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.embedding_model = XLNetModel.from_pretrained('xlnet-base-cased')
        self.experiences = []
        self.experiences_replay_times = 3
        self.loss_history = []

        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, obs, question, answer):
        # Encode text
        c_q_pair = []
        c_a_pair = []
        for i in range(self.top_k):
            c_q_pair.append(T.tensor([self.tokenizer.encode(obs + ' [SEP] ' + question[i], add_special_tokens=True)]))
            c_a_pair.append(T.tensor([self.tokenizer.encode(obs + ' [SEP] ' + answer[i], add_special_tokens=True)]))

        encoded_c_q_pair = []
        encoded_c_a_pair = []
        with T.no_grad():
            for pair in c_q_pair:
                encoded_c_q_pair.append(self.embedding_model(pair)[0][0][0])
            for pair in c_a_pair:
                encoded_c_a_pair.append(self.embedding_model(pair)[0][0][0])
        
        encoded_obs_q = encoded_c_q_pair[0]
        encoded_obs_a = encoded_c_a_pair[0]
        for i in range(1, self.top_k):
            encoded_obs_q = T.cat((encoded_obs_q, encoded_c_q_pair[i]), dim=0)
            encoded_obs_a = T.cat((encoded_obs_a, encoded_c_a_pair[i]), dim=0)

        encoded_state = T.cat((encoded_obs_q, encoded_obs_a), dim=0)

        if np.random.random() > self.epsilon:
            'if the random number is greater than exploration threshold, choose the action maximizing Q'
            state = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions)
            print(actions)
        else:
            'randomly choosing an action'
            action = np.random.choice(self.action_space)
            print(action)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        # save to experiences for experience replay
        if action:
            self.ask_experiences.append([state, action, reward, state_])
        else:
            self.answer_experiences.append([state, action, reward, state_])

        # sample from past experiences
        ask_exps = random.sample(self.ask_experiences, min(self.experiences_replay_times, len(self.ask_experiences)))
        answer_exps = random.sample(self.answer_experiences, min(self.experiences_replay_times, len(self.answer_experiences)))
        exps = ask_exps + answer_exps
        exps.append([state, action, reward, state_])

        for exp in exps:
            state, action, reward, state_ = exp[0], exp[1], exp[2], exp[3]

            obs, question, answer = state[0], state[1], state[2]
            obs_, question_, answer_ = state_[0], state_[1], state_[2]

            c_q_pair = []
            c_a_pair = []
            for i in range(self.top_k):
                c_q_pair.append(T.tensor([self.tokenizer.encode(obs + ' [SEP] ' + question[i], add_special_tokens=True)]))
                c_a_pair.append(T.tensor([self.tokenizer.encode(obs + ' [SEP] ' + answer[i], add_special_tokens=True)]))
            encoded_c_q_pair = []
            encoded_c_a_pair = []
            with T.no_grad():
                for pair in c_q_pair:
                    encoded_c_q_pair.append(self.embedding_model(pair)[0][0][0])
                for pair in c_a_pair:
                    encoded_c_a_pair.append(self.embedding_model(pair)[0][0][0])
            encoded_obs_q = encoded_c_q_pair[0]
            encoded_obs_a = encoded_c_a_pair[0]
            for i in range(1, self.top_k):
                encoded_obs_q = T.cat((encoded_obs_q, encoded_c_q_pair[i]), dim=0)
                encoded_obs_a = T.cat((encoded_obs_a, encoded_c_a_pair[i]), dim=0)
            encoded_state = T.cat((encoded_obs_q, encoded_obs_a), dim=0)

            encoded_state_ = None
            if question_ is not None and answer_ is not None:
                c_q_pair_ = []
                c_a_pair_ = []
                for i in range(self.top_k):
                    c_q_pair_.append(T.tensor([self.tokenizer.encode(obs_ + ' [SEP] ' + question_[i], add_special_tokens=True)]))
                    c_a_pair_.append(T.tensor([self.tokenizer.encode(obs_ + ' [SEP] ' + answer_[i], add_special_tokens=True)]))

                encoded_c_q_pair_ = []
                encoded_c_a_pair_ = []
                with T.no_grad():
                    for pair in c_q_pair_:
                        encoded_c_q_pair_.append(self.embedding_model(pair)[0][0][0])
                    for pair in c_a_pair_:
                        encoded_c_a_pair_.append(self.embedding_model(pair)[0][0][0])
                
                encoded_obs_q_ = encoded_c_q_pair_[0]
                encoded_obs_a_ = encoded_c_a_pair_[0]
                for i in range(1, self.top_k):
                    encoded_obs_q_ = T.cat((encoded_obs_q_, encoded_c_q_pair_[i]), dim=0)
                    encoded_obs_a_ = T.cat((encoded_obs_a_, encoded_c_a_pair_[i]), dim=0)

                encoded_state_ = T.cat((encoded_obs_q_, encoded_obs_a_), dim=0)

            self.Q.optimizer.zero_grad()
            states = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
            actions = T.tensor(action).to(self.Q.device)
            rewards = T.tensor(reward).to(self.Q.device)
            states_ = T.tensor(encoded_state_, dtype=T.float).to(self.Q.device) if encoded_state_ is not None else None

            pred = self.Q.forward(states)
            q_pred = pred[actions]
            q_next = self.Q.forward(states_).max() if encoded_state_ is not None else T.tensor(0).to(self.Q.device)
            q_target = rewards + self.gamma*q_next if encoded_state_ is not None else rewards
    

            loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
            self.loss_history.append(loss.item())
            loss.backward()
            self.Q.optimizer.step()     

        self.decrement_epsilon()

    def joint_learn(self, state, a_reward, q_reward, state_):
        # save to experiences for experience replay
        
        self.experiences.append([state, a_reward, q_reward, state_])
        
        if a_reward < q_reward:
            for da in range(self.data_augment):
                self.experiences.append([state, a_reward, q_reward, state_])
        
        # sample from past experiences
        exps = random.sample(self.experiences, min(self.experiences_replay_times, len(self.experiences)))
        exps.append([state, a_reward, q_reward, state_])

        for exp in exps:
            state, a_reward, q_reward, state_ = exp[0], exp[1], exp[2], exp[3]

            obs, question, answer = state[0], state[1], state[2]
            obs_, question_, answer_ = state_[0], state_[1], state_[2]

            c_q_pair = []
            c_a_pair = []
            for i in range(self.top_k):
                c_q_pair.append(T.tensor([self.tokenizer.encode(obs + ' [SEP] ' + question[i], add_special_tokens=True)]))
                c_a_pair.append(T.tensor([self.tokenizer.encode(obs + ' [SEP] ' + answer[i], add_special_tokens=True)]))
            encoded_c_q_pair = []
            encoded_c_a_pair = []
            with T.no_grad():
                for pair in c_q_pair:
                    encoded_c_q_pair.append(self.embedding_model(pair)[0][0][0])
                for pair in c_a_pair:
                    encoded_c_a_pair.append(self.embedding_model(pair)[0][0][0])
            encoded_obs_q = encoded_c_q_pair[0]
            encoded_obs_a = encoded_c_a_pair[0]
            for i in range(1, self.top_k):
                encoded_obs_q = T.cat((encoded_obs_q, encoded_c_q_pair[i]), dim=0)
                encoded_obs_a = T.cat((encoded_obs_a, encoded_c_a_pair[i]), dim=0)
            encoded_state = T.cat((encoded_obs_q, encoded_obs_a), dim=0)

            encoded_state_ = None
            if question_ is not None and answer_ is not None:
                c_q_pair_ = []
                c_a_pair_ = []
                for i in range(self.top_k):
                    c_q_pair_.append(T.tensor([self.tokenizer.encode(obs_ + ' [SEP] ' + question_[i], add_special_tokens=True)]))
                    c_a_pair_.append(T.tensor([self.tokenizer.encode(obs_ + ' [SEP] ' + answer_[i], add_special_tokens=True)]))

                encoded_c_q_pair_ = []
                encoded_c_a_pair_ = []
                with T.no_grad():
                    for pair in c_q_pair_:
                        encoded_c_q_pair_.append(self.embedding_model(pair)[0][0][0])
                    for pair in c_a_pair_:
                        encoded_c_a_pair_.append(self.embedding_model(pair)[0][0][0])
                
                encoded_obs_q_ = encoded_c_q_pair_[0]
                encoded_obs_a_ = encoded_c_a_pair_[0]
                for i in range(1, self.top_k):
                    encoded_obs_q_ = T.cat((encoded_obs_q_, encoded_c_q_pair_[i]), dim=0)
                    encoded_obs_a_ = T.cat((encoded_obs_a_, encoded_c_a_pair_[i]), dim=0)

                encoded_state_ = T.cat((encoded_obs_q_, encoded_obs_a_), dim=0)

            self.Q.optimizer.zero_grad()
            states = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
            a_rewards = T.tensor(a_reward).to(self.Q.device)
            q_rewards = T.tensor(q_reward).to(self.Q.device)
            states_ = T.tensor(encoded_state_, dtype=T.float).to(self.Q.device) if encoded_state_ is not None else None

            pred = self.Q.forward(states)
            q_next = self.Q.forward(states_).max() if encoded_state_ is not None else T.tensor(0).to(self.Q.device)
            q_target = T.tensor([a_rewards, q_rewards + self.gamma*q_next]).to(self.Q.device) if encoded_state_ is not None else T.tensor([a_rewards, q_rewards]).to(self.Q.device)

            loss = self.Q.loss(q_target, pred).to(self.Q.device)
            self.loss_history.append(loss.item())
            print('target', q_target, 'pred', pred, 'loss', loss.item())

            loss.backward()
            self.Q.optimizer.step()     
            '''
            pred = self.Q.forward(states)
            q_next = self.Q.forward(states_).max() if encoded_state_ is not None else T.tensor(0).to(self.Q.device)
            q_target = T.tensor([a_rewards, q_rewards + self.gamma*q_next]).to(self.Q.device) if encoded_state_ is not None else T.tensor([a_rewards, q_rewards]).to(self.Q.device)
            loss = self.Q.loss(q_target, pred).to(self.Q.device)
            print('target', q_target, 'pred', pred, 'loss', loss.item())
            '''
        self.decrement_epsilon()
