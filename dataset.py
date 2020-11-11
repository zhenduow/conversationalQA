import json
import csv
import re
import glob

class ConversationDataset():
    '''
    The conversation database class. The dataset should consist of a dictionary with the conversation identifiers as keys,
    and lists of utterances as values.
    '''
    def __init__(self, path_to_dataset, batch_size):
        self.batches = []
        self.max_len = 512
        all_data_list = glob.glob(path_to_dataset + '*')
        all_data_list.sort()
        files_in_batch = 0
        for data_file in all_data_list:
            f = open(data_file)
            data = f.readlines()
            data = [d.strip() for d in data]
            data_id = data_file.split('/')[-1]
            if files_in_batch == 0:
                self.batches.append({'conversations':{}, 'responses_pool':[], 'answers_pool':[]})
            
            self.batches[-1]['conversations'][data_id] = data
            for ut_num in range(len(data)):
                if ut_num % 2 and ut_num != (len(data) - 1) :
                    self.batches[-1]['responses_pool'].append(data[ut_num])
            self.batches[-1]['answers_pool'].append(data[-1])
            files_in_batch += 1
            if files_in_batch == batch_size:
                files_in_batch = 0

'''
def processing(input_text, max_len):
    result = re.sub('\s\s+', ' ', input_text) # replace more than one whitespace with one.
    result = re.sub(r'http\S+', '', result, flags=re.MULTILINE) # remove URL
    result = result[:max_len]
    return result.strip()
'''
