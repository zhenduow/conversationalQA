from user import User
from dataset import ConversationDataset
import logging

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    dataset = ConversationDataset('./data/MSDialog-Intent.json')
    user = User(dataset, 2, 5)
    print(len(user.dataset))
    print(user.dataset['96'])
    user.simulate('16')