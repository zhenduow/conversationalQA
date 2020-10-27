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
    def __init__(self, lr, lr_decay, weight_decay, n_actions, input_dims, hidden_size = 16):
        super(LinearDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay = weight_decay)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        score = self.fc2(hidden)

        return score

class LinearDeepNetwork(nn.Module):
    '''
    The linear deep network used by the agent.
    '''
    def __init__(self, lr, lr_decay, weight_decay, n_actions, input_dims, hidden_size = 16):
        super(LinearDeepNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay = weight_decay)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        score = F.softmax(self.fc2(hidden))

        return score


class Agent():
    '''
    The conversational QA agent.
    '''
    def __init__(self, input_dims, n_actions, lr, gamma=0.25, lr_decay = 1e-10, weight_decay = 1e-3,
                 epsilon=1.0, eps_dec=1e-3, eps_min=0.01, top_k = 1, data_augment = 10):
        self.lr = lr
        self.lr_decay = lr_decay
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.weight_decay = weight_decay 
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

        self.Q = LinearDeepQNetwork(self.lr, self.lr_decay, self.weight_decay, self.n_actions, self.input_dims)

    def choose_action(self, obs, question, answer, question_scores, answer_scores):
        question_scores = T.tensor(question_scores)
        answer_scores = T.tensor(answer_scores)
        # Encode text
        
        q_list = []
        a_list = []
        c_q_pair = []
        c_a_pair = []
        obs_query, obs_context = '[SEP]'.join(obs.split('[SEP]')[:2]), '[SEP]'.join(obs.split('[SEP]')[2:])
        tensor_query = T.tensor([self.tokenizer.encode(obs_query, add_special_tokens=True)])
        tensor_context = T.tensor([self.tokenizer.encode(obs_context, add_special_tokens=True)])
        a_list.append(T.tensor([self.tokenizer.encode(answer[0], add_special_tokens=True)]))
        for i in range(self.top_k):
            q_list.append(T.tensor([self.tokenizer.encode(question[i], add_special_tokens=True)]))
            #a_list.append(T.tensor([self.tokenizer.encode(answer[i], add_special_tokens=True)]))
            #c_q_pair.append(T.tensor([self.tokenizer.encode(obs + ' [SEP] ' + question[i], add_special_tokens=True)]))
            #c_a_pair.append(T.tensor([self.tokenizer.encode(obs + ' [SEP] ' + answer[i], add_special_tokens=True)]))

        encoded_q_list = []
        encoded_a_list = []
        encoded_c_q_pair = []
        encoded_c_a_pair = []
        
        with T.no_grad():
            encoded_query = self.embedding_model(tensor_query)[0][0][0]
            encoded_context = self.embedding_model(tensor_context)[0][0][0]
            for q in q_list:
                encoded_q_list.append(self.embedding_model(q)[0][0][0])
            for a in a_list:
                encoded_a_list.append(self.embedding_model(a)[0][0][0])
            '''
            for pair in c_q_pair:
                encoded_c_q_pair.append(self.embedding_model(pair)[0][0][0])
            for pair in c_a_pair:
                encoded_c_a_pair.append(self.embedding_model(pair)[0][0][0])
            '''
        
        encoded_q = encoded_q_list[0]
        encoded_a = encoded_a_list[0]
        #encoded_obs_q = encoded_c_q_pair[0]
        #encoded_obs_a = encoded_c_a_pair[0]
        for i in range(1, self.top_k):
            encoded_q = T.cat((encoded_q, encoded_q_list[i]), dim=0)
            #encoded_a = T.cat((encoded_a, encoded_a_list[i]), dim=0)
            #encoded_obs_q = T.cat((encoded_obs_q, encoded_c_q_pair[i]), dim=0)
            #encoded_obs_a = T.cat((encoded_obs_a, encoded_c_a_pair[i]), dim=0)
            

        encoded_state = T.cat((encoded_query, encoded_context), dim=0)
        encoded_state = T.cat((encoded_state, encoded_q), dim=0)
        encoded_state = T.cat((encoded_state, encoded_a), dim=0)
        #encoded_state = T.cat((encoded_state, encoded_obs_q), dim=0)
        #encoded_state = T.cat((encoded_state, encoded_obs_a), dim=0)
        encoded_state = T.cat((encoded_state, question_scores[:self.top_k]), dim=0)
        encoded_state = T.cat((encoded_state, answer_scores[:1]), dim=0)
        
        #encoded_state = T.cat((question_scores[:self.top_k], answer_scores[:self.top_k]), dim=0)

        if np.random.random() > self.epsilon:
            'if the random number is greater than exploration threshold, choose the action maximizing Q'
            state = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            #print(actions)
            action = T.argmax(actions).item()
        else:
            'randomly choosing an action'
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


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

            obs, question, answer, question_scores, answer_scores = state[0], state[1], state[2], state[3], state[4]
            obs_, question_, answer_, question_scores_, answer_scores_ = state_[0], state_[1], state_[2], state_[3], state_[4]

            
            question_scores = T.tensor(question_scores)
            answer_scores = T.tensor(answer_scores)
            if question_scores_ is not None:
                question_scores_ = T.tensor(question_scores_)
            if answer_scores_ is not None:
                answer_scores_ = T.tensor(answer_scores_)
            q_list = []
            a_list = []
            c_q_pair = []
            c_a_pair = []
            obs_query, obs_context = '[SEP]'.join(obs.split('[SEP]')[:2]), '[SEP]'.join(obs.split('[SEP]')[2:])
            tensor_query = T.tensor([self.tokenizer.encode(obs_query, add_special_tokens=True)])
            tensor_context = T.tensor([self.tokenizer.encode(obs_context, add_special_tokens=True)])
            a_list.append(T.tensor([self.tokenizer.encode(answer[0], add_special_tokens=True)]))
            for i in range(self.top_k): 
                q_list.append(T.tensor([self.tokenizer.encode(question[i], add_special_tokens=True)]))
                #a_list.append(T.tensor([self.tokenizer.encode(answer[i], add_special_tokens=True)]))
                #c_q_pair.append(T.tensor([self.tokenizer.encode(obs + ' [SEP] ' + question[i], add_special_tokens=True)]))
                #c_a_pair.append(T.tensor([self.tokenizer.encode(obs + ' [SEP] ' + answer[i], add_special_tokens=True)]))

            encoded_q_list = []
            encoded_a_list = []
            encoded_c_q_pair = []
            encoded_c_a_pair = []
            with T.no_grad():
                encoded_query = self.embedding_model(tensor_query)[0][0][0]
                encoded_context = self.embedding_model(tensor_context)[0][0][0]
                for q in q_list:
                    encoded_q_list.append(self.embedding_model(q)[0][0][0])
                for a in a_list:
                    encoded_a_list.append(self.embedding_model(a)[0][0][0])
                '''
                for pair in c_q_pair:
                    encoded_c_q_pair.append(self.embedding_model(pair)[0][0][0])
                for pair in c_a_pair:
                    encoded_c_a_pair.append(self.embedding_model(pair)[0][0][0])
                '''

            encoded_q = encoded_q_list[0]
            encoded_a = encoded_a_list[0]
            #encoded_obs_q = encoded_c_q_pair[0]
            #encoded_obs_a = encoded_c_a_pair[0]
            for i in range(1, self.top_k):
                encoded_q = T.cat((encoded_q, encoded_q_list[i]), dim=0)
                #encoded_a = T.cat((encoded_a, encoded_a_list[i]), dim=0)
                #encoded_obs_q = T.cat((encoded_obs_q, encoded_c_q_pair[i]), dim=0)
                #encoded_obs_a = T.cat((encoded_obs_a, encoded_c_a_pair[i]), dim=0)

            encoded_state = T.cat((encoded_query, encoded_context), dim=0)
            encoded_state = T.cat((encoded_state, encoded_q), dim=0)
            encoded_state = T.cat((encoded_state, encoded_a), dim=0)
            #encoded_state = T.cat((encoded_state, encoded_obs_q), dim=0)
            #encoded_state = T.cat((encoded_state, encoded_obs_a), dim=0)
            encoded_state = T.cat((encoded_state, question_scores[:self.top_k]), dim=0)
            encoded_state = T.cat((encoded_state, answer_scores[:1]), dim=0) 


            encoded_state_ = None
            if question_ is not None and answer_ is not None:
                q_list_ = []
                a_list_ = []
                c_q_pair_ = []
                c_a_pair_ = []
                obs_query_, obs_context_ = '[SEP]'.join(obs_.split('[SEP]')[:2]), '[SEP]'.join(obs_.split('[SEP]')[2:])
                tensor_query_ = T.tensor([self.tokenizer.encode(obs_query_, add_special_tokens=True)])
                tensor_context_ = T.tensor([self.tokenizer.encode(obs_context_, add_special_tokens=True)])                    
                a_list_.append(T.tensor([self.tokenizer.encode(answer_[0], add_special_tokens=True)]))
                for i in range(self.top_k):
                    q_list_.append(T.tensor([self.tokenizer.encode(question_[i], add_special_tokens=True)]))
                    #a_list_.append(T.tensor([self.tokenizer.encode(answer_[i], add_special_tokens=True)]))
                    #c_q_pair_.append(T.tensor([self.tokenizer.encode(obs_ + ' [SEP] ' + question_[i], add_special_tokens=True)]))
                    #c_a_pair_.append(T.tensor([self.tokenizer.encode(obs_ + ' [SEP] ' + answer_[i], add_special_tokens=True)]))

                encoded_q_list_ = []
                encoded_a_list_ = []
                encoded_c_q_pair_ = []
                encoded_c_a_pair_ = []
                with T.no_grad():
                    encoded_query_ = self.embedding_model(tensor_query_)[0][0][0]
                    encoded_context_ = self.embedding_model(tensor_context_)[0][0][0]
                    for q in q_list:
                        encoded_q_list_.append(self.embedding_model(q)[0][0][0])
                    for a in a_list:
                        encoded_a_list_.append(self.embedding_model(a)[0][0][0])
                    '''
                    for pair in c_q_pair_:
                        encoded_c_q_pair_.append(self.embedding_model(pair)[0][0][0])
                    for pair in c_a_pair_:
                        encoded_c_a_pair_.append(self.embedding_model(pair)[0][0][0])
                    '''

                encoded_q_ = encoded_q_list_[0]
                encoded_a_ = encoded_a_list_[0]
                #encoded_obs_q_ = encoded_c_q_pair_[0]
                #encoded_obs_a_ = encoded_c_a_pair_[0]
                for i in range(1, self.top_k):
                    encoded_q_ = T.cat((encoded_q_, encoded_q_list_[i]), dim=0)
                    #encoded_a_ = T.cat((encoded_a_, encoded_a_list_[i]), dim=0)
                    #encoded_obs_q_ = T.cat((encoded_obs_q_, encoded_c_q_pair_[i]), dim=0)
                    #encoded_obs_a_ = T.cat((encoded_obs_a_, encoded_c_a_pair_[i]), dim=0)
                
                encoded_state_ = T.cat((encoded_query_, encoded_context_), dim=0)
                encoded_state_ = T.cat((encoded_state_, encoded_q_), dim=0)
                encoded_state_ = T.cat((encoded_state_, encoded_a_), dim=0)
                #encoded_state_ = T.cat((encoded_state_, encoded_obs_q_), dim=0)
                #encoded_state_ = T.cat((encoded_state_, encoded_obs_a_), dim=0)
                encoded_state_ = T.cat((encoded_state_, question_scores_[:self.top_k]), dim=0)
                encoded_state_ = T.cat((encoded_state_, answer_scores_[:1]), dim=0)
            
            #encoded_state = T.cat((question_scores[:self.top_k], answer_scores[:self.top_k]), dim=0)
            #encoded_state_ = T.cat((question_scores_[:self.top_k], answer_scores_[:self.top_k]), dim=0)

            self.Q.optimizer.zero_grad()
            states = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
            a_rewards = T.tensor(a_reward).to(self.Q.device)
            q_rewards = T.tensor(q_reward).to(self.Q.device)
            states_ = T.tensor(encoded_state_, dtype=T.float).to(self.Q.device) if encoded_state_ is not None else None

            pred = self.Q.forward(states)
            q_next = self.Q.forward(states_).max() if encoded_state_ is not None else T.tensor(0).to(self.Q.device)
            q_target = T.tensor([a_rewards, q_rewards + self.gamma*q_next]).to(self.Q.device) if encoded_state_ is not None else T.tensor([a_rewards, q_rewards]).to(self.Q.device)


            loss = self.Q.loss(q_target, pred).to(self.Q.device)
            
            # l1 penalty
            l1 = 0
            for p in self.Q.parameters():
                l1 += p.abs().sum()
            
            loss = loss + self.weight_decay * l1
            
            self.loss_history.append(loss.item())
            #print('target', q_target, 'pred', pred, 'loss', loss.item())

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


class BaseAgent():
    '''
    The Baseline conversational QA agent.
    '''
    def __init__(self, input_dims, n_actions, lr, lr_decay = 1e-10, weight_decay = 1e-3):
        self.lr = lr
        self.lr_decay = lr_decay
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.weight_decay = weight_decay 
        
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.embedding_model = XLNetModel.from_pretrained('xlnet-base-cased')
        self.loss_history = []

        self.Q = LinearDeepNetwork(self.lr, self.lr_decay, self.weight_decay, self.n_actions, self.input_dims)

    def choose_action(self, obs):
        
        obs_query, obs_context = '[SEP]'.join(obs.split('[SEP]')[:2]), '[SEP]'.join(obs.split('[SEP]')[2:])
        tensor_query = T.tensor([self.tokenizer.encode(obs_query, add_special_tokens=True)])
        tensor_context = T.tensor([self.tokenizer.encode(obs_context, add_special_tokens=True)])
        
        with T.no_grad():
            encoded_query = self.embedding_model(tensor_query)[0][0][0]
            encoded_context = self.embedding_model(tensor_context)[0][0][0]

        encoded_state = T.cat((encoded_query, encoded_context), dim=0)

        
        state = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
        actions = self.Q.forward(state)
        action = T.argmax(actions).item()
        
        return action

    def learn(self, obs, true_label):
        # save to experiences for experience replay
        
        
        obs_query, obs_context = '[SEP]'.join(obs.split('[SEP]')[:2]), '[SEP]'.join(obs.split('[SEP]')[2:])
        tensor_query = T.tensor([self.tokenizer.encode(obs_query, add_special_tokens=True)])
        tensor_context = T.tensor([self.tokenizer.encode(obs_context, add_special_tokens=True)])
           
        with T.no_grad():
            encoded_query = self.embedding_model(tensor_query)[0][0][0]
            encoded_context = self.embedding_model(tensor_context)[0][0][0]
               
        encoded_state = T.cat((encoded_query, encoded_context), dim=0)
       

        self.Q.optimizer.zero_grad()
        states = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
        
        pred = self.Q.forward(states)
        q_target = T.tensor([1, 0]).to(self.Q.device) if true_label == 0 else T.tensor([0, 1]).to(self.Q.device)
        loss = self.Q.loss(q_target, pred).to(self.Q.device)
            
        # l1 penalty
        l1 = 0
        for p in self.Q.parameters():
            l1 += p.abs().sum()
            
        loss = loss + self.weight_decay * l1
            
        self.loss_history.append(loss.item())
        print('target', q_target, 'pred', pred, 'loss', loss.item())

        loss.backward()
        self.Q.optimizer.step()     
            


class ScoreAgent():
    '''
    using only the ranking scores.
    '''
    def __init__(self, input_dims, n_actions, lr, gamma=0.25, lr_decay = 1e-10, weight_decay = 1e-3,
                 epsilon=1.0, eps_dec=1e-3, eps_min=0.01, top_k = 1, data_augment = 10):
        self.lr = lr
        self.lr_decay = lr_decay
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.weight_decay = weight_decay 
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.top_k = top_k
        self.data_augment = data_augment
        self.action_space = [i for i in range(self.n_actions)]
        self.experiences = []
        self.experiences_replay_times = 3
        self.loss_history = []

        self.Q = LinearDeepQNetwork(self.lr, self.lr_decay, self.weight_decay, self.n_actions, self.input_dims)

    def choose_action(self, question_scores, answer_scores):
        question_scores = T.tensor(question_scores)
        answer_scores = T.tensor(answer_scores)
        encoded_state = T.cat((question_scores[:self.top_k], answer_scores[:1]), dim=0)

        if np.random.random() > self.epsilon:
            'if the random number is greater than exploration threshold, choose the action maximizing Q'
            state = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            #print(actions)
            action = T.argmax(actions).item()
        else:
            'randomly choosing an action'
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


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

            obs, question, answer, question_scores, answer_scores = state[0], state[1], state[2], state[3], state[4]
            obs_, question_, answer_, question_scores_, answer_scores_ = state_[0], state_[1], state_[2], state_[3], state_[4]

            
            question_scores = T.tensor(question_scores)
            answer_scores = T.tensor(answer_scores)
            if question_scores_ is not None:
                question_scores_ = T.tensor(question_scores_)
            if answer_scores_ is not None:
                answer_scores_ = T.tensor(answer_scores_)

            encoded_state = T.cat((question_scores[:self.top_k], answer_scores[:1]), dim=0)
            if question_scores_ is not None and answer_scores_ is not None:
                encoded_state_ = T.cat((question_scores_[:self.top_k], answer_scores_[:1]), dim=0)
            else:
                encoded_state_ = None

            self.Q.optimizer.zero_grad()
            states = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
            a_rewards = T.tensor(a_reward).to(self.Q.device)
            q_rewards = T.tensor(q_reward).to(self.Q.device)
            states_ = T.tensor(encoded_state_, dtype=T.float).to(self.Q.device) if encoded_state_ is not None else None

            pred = self.Q.forward(states)
            q_next = self.Q.forward(states_).max() if encoded_state_ is not None else T.tensor(0).to(self.Q.device)
            q_target = T.tensor([a_rewards, q_rewards + self.gamma*q_next]).to(self.Q.device) if encoded_state_ is not None else T.tensor([a_rewards, q_rewards]).to(self.Q.device)


            loss = self.Q.loss(q_target, pred).to(self.Q.device)
            
            # l1 penalty
            l1 = 0
            for p in self.Q.parameters():
                l1 += p.abs().sum()
            
            loss = loss + self.weight_decay * l1
            
            self.loss_history.append(loss.item())
            #print('target', q_target, 'pred', pred, 'loss', loss.item())

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



class TextAgent():
    '''
    Using only the encoded text.
    '''
    def __init__(self, input_dims, n_actions, lr, gamma=0.25, lr_decay = 1e-10, weight_decay = 1e-3,
                 epsilon=1.0, eps_dec=1e-3, eps_min=0.01, top_k = 1, data_augment = 10):
        self.lr = lr
        self.lr_decay = lr_decay
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.weight_decay = weight_decay 
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

        self.Q = LinearDeepQNetwork(self.lr, self.lr_decay, self.weight_decay, self.n_actions, self.input_dims)

    def choose_action(self, obs, question, answer):
        # Encode text
        
        q_list = []
        a_list = []
        c_q_pair = []
        c_a_pair = []
        obs_query, obs_context = '[SEP]'.join(obs.split('[SEP]')[:2]), '[SEP]'.join(obs.split('[SEP]')[2:])
        tensor_query = T.tensor([self.tokenizer.encode(obs_query, add_special_tokens=True)])
        tensor_context = T.tensor([self.tokenizer.encode(obs_context, add_special_tokens=True)])
        a_list.append(T.tensor([self.tokenizer.encode(answer[0], add_special_tokens=True)]))
        for i in range(self.top_k):
            q_list.append(T.tensor([self.tokenizer.encode(question[i], add_special_tokens=True)]))

        encoded_q_list = []
        encoded_a_list = []
        encoded_c_q_pair = []
        encoded_c_a_pair = []
        
        with T.no_grad():
            encoded_query = self.embedding_model(tensor_query)[0][0][0]
            encoded_context = self.embedding_model(tensor_context)[0][0][0]
            for q in q_list:
                encoded_q_list.append(self.embedding_model(q)[0][0][0])
            for a in a_list:
                encoded_a_list.append(self.embedding_model(a)[0][0][0])
        
        encoded_q = encoded_q_list[0]
        encoded_a = encoded_a_list[0]
        for i in range(1, self.top_k):
            encoded_q = T.cat((encoded_q, encoded_q_list[i]), dim=0)
            

        encoded_state = T.cat((encoded_query, encoded_context), dim=0)
        encoded_state = T.cat((encoded_state, encoded_q), dim=0)
        encoded_state = T.cat((encoded_state, encoded_a), dim=0)
        

        if np.random.random() > self.epsilon:
            'if the random number is greater than exploration threshold, choose the action maximizing Q'
            state = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            #print(actions)
            action = T.argmax(actions).item()
        else:
            'randomly choosing an action'
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


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

            obs, question, answer, question_scores, answer_scores = state[0], state[1], state[2], state[3], state[4]
            obs_, question_, answer_, question_scores_, answer_scores_ = state_[0], state_[1], state_[2], state_[3], state_[4]

            q_list = []
            a_list = []
            c_q_pair = []
            c_a_pair = []
            obs_query, obs_context = '[SEP]'.join(obs.split('[SEP]')[:2]), '[SEP]'.join(obs.split('[SEP]')[2:])
            tensor_query = T.tensor([self.tokenizer.encode(obs_query, add_special_tokens=True)])
            tensor_context = T.tensor([self.tokenizer.encode(obs_context, add_special_tokens=True)])
            a_list.append(T.tensor([self.tokenizer.encode(answer[0], add_special_tokens=True)]))
            for i in range(self.top_k): 
                q_list.append(T.tensor([self.tokenizer.encode(question[i], add_special_tokens=True)]))

            encoded_q_list = []
            encoded_a_list = []
            encoded_c_q_pair = []
            encoded_c_a_pair = []
            with T.no_grad():
                encoded_query = self.embedding_model(tensor_query)[0][0][0]
                encoded_context = self.embedding_model(tensor_context)[0][0][0]
                for q in q_list:
                    encoded_q_list.append(self.embedding_model(q)[0][0][0])
                for a in a_list:
                    encoded_a_list.append(self.embedding_model(a)[0][0][0])

            encoded_q = encoded_q_list[0]
            encoded_a = encoded_a_list[0]
            for i in range(1, self.top_k):
                encoded_q = T.cat((encoded_q, encoded_q_list[i]), dim=0)

            encoded_state = T.cat((encoded_query, encoded_context), dim=0)
            encoded_state = T.cat((encoded_state, encoded_q), dim=0)
            encoded_state = T.cat((encoded_state, encoded_a), dim=0)

            encoded_state_ = None
            if question_ is not None and answer_ is not None:
                q_list_ = []
                a_list_ = []
                c_q_pair_ = []
                c_a_pair_ = []
                obs_query_, obs_context_ = '[SEP]'.join(obs_.split('[SEP]')[:2]), '[SEP]'.join(obs_.split('[SEP]')[2:])
                tensor_query_ = T.tensor([self.tokenizer.encode(obs_query_, add_special_tokens=True)])
                tensor_context_ = T.tensor([self.tokenizer.encode(obs_context_, add_special_tokens=True)])                    
                a_list_.append(T.tensor([self.tokenizer.encode(answer_[0], add_special_tokens=True)]))
                for i in range(self.top_k):
                    q_list_.append(T.tensor([self.tokenizer.encode(question_[i], add_special_tokens=True)]))

                encoded_q_list_ = []
                encoded_a_list_ = []
                encoded_c_q_pair_ = []
                encoded_c_a_pair_ = []
                with T.no_grad():
                    encoded_query_ = self.embedding_model(tensor_query_)[0][0][0]
                    encoded_context_ = self.embedding_model(tensor_context_)[0][0][0]
                    for q in q_list:
                        encoded_q_list_.append(self.embedding_model(q)[0][0][0])
                    for a in a_list:
                        encoded_a_list_.append(self.embedding_model(a)[0][0][0])

                encoded_q_ = encoded_q_list_[0]
                encoded_a_ = encoded_a_list_[0]
                for i in range(1, self.top_k):
                    encoded_q_ = T.cat((encoded_q_, encoded_q_list_[i]), dim=0)
                
                encoded_state_ = T.cat((encoded_query_, encoded_context_), dim=0)
                encoded_state_ = T.cat((encoded_state_, encoded_q_), dim=0)
                encoded_state_ = T.cat((encoded_state_, encoded_a_), dim=0)
            

            self.Q.optimizer.zero_grad()
            states = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
            a_rewards = T.tensor(a_reward).to(self.Q.device)
            q_rewards = T.tensor(q_reward).to(self.Q.device)
            states_ = T.tensor(encoded_state_, dtype=T.float).to(self.Q.device) if encoded_state_ is not None else None

            pred = self.Q.forward(states)
            q_next = self.Q.forward(states_).max() if encoded_state_ is not None else T.tensor(0).to(self.Q.device)
            q_target = T.tensor([a_rewards, q_rewards + self.gamma*q_next]).to(self.Q.device) if encoded_state_ is not None else T.tensor([a_rewards, q_rewards]).to(self.Q.device)


            loss = self.Q.loss(q_target, pred).to(self.Q.device)
            
            # l1 penalty
            l1 = 0
            for p in self.Q.parameters():
                l1 += p.abs().sum()
            
            loss = loss + self.weight_decay * l1
            
            self.loss_history.append(loss.item())
            #print('target', q_target, 'pred', pred, 'loss', loss.item())

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
