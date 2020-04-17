import json
import csv
import re

class ConversationDataset():
    '''
    The conversation database class. The dataset should consist of a dictionary with the conversation identifiers as keys,
    and lists of utterances as values.
    '''
    def __init__(self, path_to_dataset):
        with open(path_to_dataset) as f:
            data = json.load(f)
        self.conversations = {}

        for k in data.keys():
            participants = []
            intents = []
            for utterance in data[k]['utterances']:
                participants.append(utterance['user_id'])
                intents.extend(utterance['tags'].split())
            if set(['CQ', 'IR', 'FD']) & set(intents):
                # keep conversations which contains any form of information need.
                self.conversations[k] = []
                role = 'User'
                for utterance in data[k]['utterances']:
                    if utterance['tags'] == 'GG' or utterance['tags'] == 'JK' or utterance['tags'] == 'GG JK':
                        continue
                    unprocessed = utterance['utterance']
                    processed = self.processing(unprocessed)
                    if self.conversations[k] == []:
                        self.conversations[k] = [processed]
                    elif utterance['actor_type'] == role:
                        # concatenate all consecutive utterances from one role together
                        self.conversations[k][-1] += '. '
                        self.conversations[k][-1] += processed
                    else:
                        self.conversations[k].append(processed)
                        role = utterance['actor_type']

    def processing(self, input_text):
        '''
        A simple processing function to preprocess the data.
        '''
        result = re.sub('\s\s+', ' ', input_text) # replace more than one whitespace with one.
        result = re.sub(r'http\S+', '', result, flags=re.MULTILINE) # remove URL
        return result
