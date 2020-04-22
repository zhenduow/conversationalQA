from user import User
from dataset import ConversationDataset
from agent import Agent
import logging
import numpy as np

observation_dim = 512
observation_num = 10
n_conversations = 10000

#if __name__ == "__main__":
#   logging.getLogger().setLevel(logging.INFO)
#    dataset = ConversationDataset('./data/MSDialog-Intent.json')
#    user = User(dataset, 2, 5)
#    user.simulate('96')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    dataset = ConversationDataset('./data/MSDialog-Intent.json')
    user = User(dataset, 2, 5)

    scores = []
    eps_history = []

    # need to define n_actions
    agent = Agent(lr=0.0001, input_dims=observation_dim,
                  n_actions=observation_num)

    for i in range(n_conversations):
        score = 0
        done = False
        # need to complete user.initialize_state
        obs = user.initialize_state(i)

        while not done:
            action = agent.choose_action(obs)
            # need to complete user.update_state
            obs_, reward, done, info = user.update_state(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.1f avg score %.1f epsilon %.2f' %
                  (score, avg_score, agent.epsilon))