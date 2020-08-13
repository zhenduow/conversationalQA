import json
import csv
import re
import glob

class ConversationDataset():
    '''
    The conversation database class. The dataset should consist of a dictionary with the conversation identifiers as keys,
    and lists of utterances as values.
    '''
    def __init__(self, path_to_dataset, dataset_size):
        self.conversations = {}
        self.responses_pool = []
        self.answers_pool = []
        self.max_len = 512
        all_data_list = glob.glob(path_to_dataset + '*')
        total = 0
        for data_file in all_data_list:
            if total > dataset_size:
                break
            
            total += 1

            f = open(data_file)
            data = f.readlines()
            data = [d.strip() for d in data]
            data_id = data_file.split('/')[3]
            self.conversations[data_id] = data

            for ut_num in range(len(data)):
                if ut_num % 2 and ut_num != (len(data) - 1) :
                    self.responses_pool.append(data[ut_num])
            
            self.answers_pool.append(data[-1])

'''
def processing(input_text, max_len):
    result = re.sub('\s\s+', ' ', input_text) # replace more than one whitespace with one.
    result = re.sub(r'http\S+', '', result, flags=re.MULTILINE) # remove URL
    result = result[:max_len]
    return result.strip()
'''
