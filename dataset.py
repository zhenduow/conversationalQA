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
        self.responses_pool = []
        self.answers_pool = []
        self.questions_pool = []
        self.max_len = 360

        for k in data.keys():
            participants = []
            intents = []
            final_answer = ''
            for utterance in data[k]['utterances']:
                participants.append(utterance['user_id'])
                intents.extend(utterance['tags'].split())
            if set(['CQ', 'IR', 'FD']) & set(intents):
                # keep conversations which contains any form of information need.
                self.conversations[k] = []
                role = 'User'
                for utterance_id, utterance in enumerate(data[k]['utterances']):
                    if set(utterance['tags'].split()).issubset(set(['GG', 'JK', 'PF'])): 
                        # remove greetings or junk messages
                        continue
                    unprocessed = utterance['utterance']
                    processed = processing(unprocessed, self.max_len)
                    if self.conversations[k] == []:
                        self.conversations[k] = [data[k]['title'] + '. [SEP] ' + processed]
                    elif utterance['actor_type'] == role:
                        # concatenate all consecutive utterances from one role together
                        self.conversations[k][-1] += '. '
                        self.conversations[k][-1] += processed
                    else:
                        if utterance['actor_type'] == 'User' and utterance_id == len(data[k]['utterances']) -1 :
                            break
                        self.conversations[k].append(processed)

                     # adding any agent response to response pool and update it as the final_answer
                    if utterance['is_answer'] == 1:
                        final_answer = processed
                        break

                    # update role
                    role = utterance['actor_type']

                if len(self.conversations[k]) %2:
                    del self.conversations[k]
                    continue

                for final_ut_id in range(len(self.conversations[k])):
                    if final_ut_id % 2 :
                        if final_ut_id != len(self.conversations[k]) - 1:
                            self.responses_pool.append(self.conversations[k][final_ut_id])
                    else:
                        self.questions_pool.append(self.conversations[k][final_ut_id])
                self.answers_pool.append(final_answer)
        
        #assert not set.intersection(set(self.responses_pool),set(self.answers_pool))

def processing(input_text, max_len):
    '''
    A simple processing function to preprocess the data.
    '''
    result = re.sub('\s\s+', ' ', input_text) # replace more than one whitespace with one.
    result = re.sub(r'http\S+', '', result, flags=re.MULTILINE) # remove URL
    result = result[:max_len]
    return result.strip()
