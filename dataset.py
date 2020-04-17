import json
import csv

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
                    unprocessed = utterance['utterance']
                    processed = self.processing(unprocessed)
                    if self.conversations[k] == []:
                        self.conversations[k] = [processed]
                    elif utterance['actor_type'] == role:
                        self.conversations[k][-1] += '. '
                        self.conversations[k][-1] += processed
                    else:
                        self.conversations[k].append(processed)
                        role = utterance['actor_type']

    def processing(self, input_text):
        '''
        A simple processing function to remove abundant white characters and greeting words like 'hi' and 'thank you'.
        '''

        result = input_text.replace('')

        return result
