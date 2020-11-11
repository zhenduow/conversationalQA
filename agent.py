import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from transformers import *
import warnings 
import random
import time
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
        self.tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
        self.embedding_model = AutoModel.from_pretrained('xlnet-base-cased')
        self.experiences = []
        self.experiences_replay_times = 3
        self.loss_history = []

        self.Q = LinearDeepQNetwork(self.lr, self.lr_decay, self.weight_decay, self.n_actions, self.input_dims)

    def choose_action(self, query_embedding, context_embedding, questions_embeddings, answers_embeddings, question_scores, answer_scores):
        encoded_q = questions_embeddings[0]
        for i in range(1, self.top_k):
            encoded_q = T.cat((encoded_q, questions_embeddings[i]), dim=0)
            
        encoded_state = T.cat((query_embedding, context_embedding), dim=0)
        encoded_state = T.cat((encoded_state, encoded_q), dim=0)
        encoded_state = T.cat((encoded_state, answers_embeddings[0]), dim=0)
        encoded_state = T.cat((encoded_state, question_scores[:self.top_k]), dim=0)
        encoded_state = T.cat((encoded_state, answer_scores[:1]), dim=0)
    
        if np.random.random() > self.epsilon:
            state = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
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

            query_embedding, context_embedding, questions_embeddings, answers_embeddings, question_scores, answer_scores = state[0], state[1], state[2], state[3], state[4], state[5]
            query_embedding, context_embedding_, questions_embeddings_, answers_embeddings_, question_scores_, answer_scores_ = state_[0], state_[1], state_[2], state_[3], state_[4], state_[5]

            encoded_q = questions_embeddings[0]
            for i in range(1, self.top_k):
                encoded_q = T.cat((encoded_q, questions_embeddings[i]), dim=0)

            encoded_state = T.cat((query_embedding, context_embedding), dim=0)
            encoded_state = T.cat((encoded_state, encoded_q), dim=0)
            encoded_state = T.cat((encoded_state, answers_embeddings[0]), dim=0)
            encoded_state = T.cat((encoded_state, question_scores[:self.top_k]), dim=0)
            encoded_state = T.cat((encoded_state, answer_scores[:1]), dim=0) 

            encoded_state_ = None
            if questions_embeddings_ is not None and answers_embeddings_ is not None:
                encoded_q_ = questions_embeddings_[0]
                for i in range(1, self.top_k):
                    encoded_q_ = T.cat((encoded_q_, questions_embeddings_[i]), dim=0)
                
                encoded_state_ = T.cat((query_embedding, context_embedding_), dim=0)
                encoded_state_ = T.cat((encoded_state_, encoded_q_), dim=0)
                encoded_state_ = T.cat((encoded_state_, answers_embeddings_[0]), dim=0)
                encoded_state_ = T.cat((encoded_state_, question_scores_[:self.top_k]), dim=0)
                encoded_state_ = T.cat((encoded_state_, answer_scores_[:1]), dim=0)
            
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
            loss.backward()
            self.Q.optimizer.step()     
            
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
        
        self.tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
        self.embedding_model = AutoModel.from_pretrained('xlnet-base-cased')
        self.loss_history = []

        self.Q = LinearDeepNetwork(self.lr, self.lr_decay, self.weight_decay, self.n_actions, self.input_dims)

    def choose_action(self, query_embedding, context_embedding):
        
        encoded_state = T.cat((query_embedding, context_embedding), dim=0)
        state = T.tensor(encoded_state, dtype=T.float).to(self.Q.device)
        actions = self.Q.forward(state)
        action = T.argmax(actions).item()
        
        return action

    def learn(self, query_embedding, context_embedding, true_label):
        # save to experiences for experience replay     
        encoded_state = T.cat((query_embedding, context_embedding), dim=0)
       
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

            question_scores, answer_scores = state[0], state[1]
            question_scores_, answer_scores_ = state_[0], state_[1]

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
            loss.backward()
            self.Q.optimizer.step()     
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
        self.tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
        self.embedding_model = AutoModel.from_pretrained('xlnet-base-cased')
        self.experiences = []
        self.experiences_replay_times = 3
        self.loss_history = []

        self.Q = LinearDeepQNetwork(self.lr, self.lr_decay, self.weight_decay, self.n_actions, self.input_dims)

    def choose_action(self, query_embedding, context_embedding, questions_embeddings, answers_embeddings):
        # Encode text
        
        encoded_q = questions_embeddings[0]
        for i in range(1, self.top_k):
            encoded_q = T.cat((encoded_q, questions_embeddings[i]), dim=0)
        encoded_state = T.cat((query_embedding, context_embedding), dim=0)
        encoded_state = T.cat((encoded_state, encoded_q), dim=0)
        encoded_state = T.cat((encoded_state, answers_embeddings[0]), dim=0)
        
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

            query_embedding, context_embedding, questions_embeddings, answers_embeddings= state[0], state[1], state[2], state[3]
            query_embedding, context_embedding_, questions_embeddings_, answers_embeddings_ = state_[0], state_[1], state_[2], state_[3]

            encoded_q = questions_embeddings[0]
            for i in range(1, self.top_k):
                encoded_q = T.cat((encoded_q, questions_embeddings[i]), dim=0)

            encoded_state = T.cat((query_embedding, context_embedding), dim=0)
            encoded_state = T.cat((encoded_state, encoded_q), dim=0)
            encoded_state = T.cat((encoded_state, answers_embeddings[0]), dim=0)

            encoded_state_ = None
            if questions_embeddings_ is not None and answers_embeddings_ is not None:

                encoded_q_ = questions_embeddings_[0]
                for i in range(1, self.top_k):
                    encoded_q_ = T.cat((encoded_q_, questions_embeddings_[i]), dim=0)
                
                encoded_state_ = T.cat((query_embedding, context_embedding_), dim=0)
                encoded_state_ = T.cat((encoded_state_, encoded_q_), dim=0)
                encoded_state_ = T.cat((encoded_state_, answers_embeddings_[0]), dim=0)
            
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
            loss.backward()
            self.Q.optimizer.step()     
        self.decrement_epsilon()
