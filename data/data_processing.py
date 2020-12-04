import json
import csv
import re
import os
import numpy as np
import resource
import glob
import random
import argparse
import copy
import shutil

product_name = set(['Windows_7', 'Skype_Android', 'Apps_Windows_10', 'Excel', 'Word',
    'Outlook_Contacts', 'Office_Insider', 'Outlook_Calendar', 'Windows_Insider_Apps',
    'Bing', 'Bing_Maps', 'Windows_RT_8.1', 'Office_Install', 'Office_Account',
    'Games_Windows_10', 'Outlook_Email', 'Outlook_Preview', 'Outlook', 'Skype_Windows_Desktop',
    'PowerPoint', 'Windows_10', 'Skype_iOS', 'Windows_Insider_Games', 'Skype_Linux',
    'Windows_Insider_IE_Edge', 'Windows_Insider_Register', 'Skype_Mac', 'Skype_Web',
    'Skype_Lite', 'Windows_Insider_Preview', 'Windows_Insider_Office', 'Bing_Apps',
    'Skype_Xbox', 'Bing_Ads', 'Bing_Search', 'Windows_8.1', 'Bing_Safety', 'Skype_Windows_10', 'Windows_Mobile'])

def processing(input_text, max_len):
    '''
    A simple processing function to preprocess the data.
    '''
    result = re.sub('\s\s+', ' ', input_text) # replace more than one whitespace with one.
    result = re.sub(r'http\S+', '', result) # remove URL
    result = ' '.join((result.split()[:max_len]))
    return result.strip()

if __name__ == "__main__":
    max_len = 512
    random.seed(2020)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type = str, default = 'MSDialog')
    parser.add_argument('--n_folds', type = int, default = 5) # only for MSDialog since it's small
    args = parser.parse_args()
    if not os.path.exists(args.dataset_name + '-Complete'):
        os.makedirs(args.dataset_name + '-Complete')
    if not os.path.exists(args.dataset_name + '-Incomplete'):
        os.makedirs(args.dataset_name + '-Incomplete')
    complete_conversations = {}
    incomplete_conversations = {}

    if args.dataset_name == 'MSDialog':
        max_data_size = 3600
        total_complete_size = 0
        with open('MSDialog-Complete.json') as f:
            data = json.load(f)
            for k in data.keys():
                #if data[k]['category'] not in product_name:
                #    continue
                is_answer_label = [1 if ut['is_answer'] == 1 else 0 for ut in data[k]['utterances']]
                has_answer = np.sum(is_answer_label)
                if has_answer > 0:
                    conversation = []
                    for utterance_id, utterance in enumerate(data[k]['utterances']):
                        unprocessed = utterance['utterance']
                        processed = processing(unprocessed, max_len).strip()
                        if conversation== []:
                            conversation = [data[k]['title'] + '. [SEP] ' + processed]
                            #conversation = [processed]
                        elif utterance['actor_type'] == role:
                            # concatenate all consecutive utterances from one role together
                            conversation[-1] += '. '
                            conversation[-1] += processed
                        else:
                            if utterance['actor_type'] == 'User' and utterance_id == len(data[k]['utterances']) -1 :
                                break
                            if processed != '':
                                conversation.append(processed)
                                
                        # adding any agent response to response pool and update it as the final_answer
                        if utterance['is_answer'] == 1:
                            final_answer = processed
                            break
                        role = utterance['actor_type']

                    for convid in range(len(conversation)):
                            conversation[convid] = processing(conversation[convid], max_len)

                    if len(conversation) >= 4 and len(conversation) <= 10 and len(conversation) % 2 == 0 and len(conversation[-1].split()) < 512 :
                        total_complete_size += 1
                        if total_complete_size <= max_data_size:
                            complete_conversations[k] = conversation

                    elif len(conversation) % 2 == 1:
                        conversation = conversation[:-1]
                        incomplete_conversations[k] = conversation
                    
                    elif len(conversation) <= 20 :
                        incomplete_conversations[k] = conversation

                else:    
                    # conversation has no answer, so take all user-agent turn pairs as query-cq pairs.
                    conversation = []
                    for utterance_id, utterance in enumerate(data[k]['utterances']):
                        unprocessed = utterance['utterance']
                        processed = processing(unprocessed  , max_len).strip()
                        if conversation== []:
                            conversation = [data[k]['title'] + '. [SEP] ' + processed]
                            #conversation = [processed]
                        elif utterance['actor_type'] == role:
                            # concatenate all consecutive utterances from one role together
                            conversation[-1] += '. '
                            conversation[-1] += processed
                        else:
                            if processed != '':
                                conversation.append(processed)
                        role = utterance['actor_type']
                    if len(conversation) % 2 == 1:
                        conversation = conversation[:-1]
                    if 2 <= len(conversation) <= 20 :
                        incomplete_conversations[k] = conversation
        
        # split data into folds and write to file
        all_data_list = list(complete_conversations.keys())
        fold_size = int(len(all_data_list)/args.n_folds)
        remaining_data_list = copy.deepcopy(all_data_list)
        folds = []
        for i in range(args.n_folds):
            folds.append(random.sample(remaining_data_list, fold_size))
            remaining_data_list = [d for d in remaining_data_list if d not in folds[i]]
        
        for i in range(args.n_folds-2):
            assert not(set(folds[i]) & set(folds[i+1]))

        for i in range(args.n_folds):
            if not os.path.exists(args.dataset_name + '-Complete/train' + str(i)):
                os.makedirs(args.dataset_name + '-Complete/train' + str(i))
            if not os.path.exists(args.dataset_name + '-Complete/test' + str(i)):
                os.makedirs(args.dataset_name + '-Complete/test' + str(i))
                
            for f in [d for d in all_data_list if d not in folds[i]]:
                with open(args.dataset_name + '-Complete/train' + str(i) + '/' + f, 'w') as output:
                    output.write('\n'.join(complete_conversations[f]))
            for f in folds[i]:
                with open(args.dataset_name + '-Complete/test' + str(i) + '/' + f, 'w') as output:
                    output.write('\n'.join(complete_conversations[f]))
        
    elif args.dataset_name == 'UDC':
        max_data_size = 10000
        total_complete_size, total_incomplete_size = 0, 0
        all_folders = glob.glob('ubuntu_raw_dialogs/*')
        conversation_serial = 0
        for folder in all_folders:
            all_files = glob.glob(folder + '/*')
            for f in all_files:
                if total_complete_size >= max_data_size and total_incomplete_size > max_data_size:
                    break
                conversation_serial += 1
                lines = open(f).readlines()
                sender_list = []
                conversation = []
                for line in lines:
                    elements = line.split('\t')
                    sender, recipient, utterance = elements[1], elements[2], processing(elements[3], max_len)
                    if len(sender_list) > 0 and sender == sender_list[-1]:
                        conversation[-1] += utterance
                    else:
                        sender_list.append(sender)
                        conversation.append(utterance)
                if len(list(set(sender_list))) <= 2 and len(conversation) >=4 and len(conversation) <= 10 and len(conversation) % 2 == 0:
                    total_complete_size += 1
                    if total_complete_size <= max_data_size:
                        complete_conversations[conversation_serial] = conversation
                elif len(conversation) >= 4:
                    total_incomplete_size += 1
                    if total_incomplete_size <= max_data_size:
                        incomplete_conversations[conversation_serial] = conversation
                    
        # write to file
        if not os.path.exists(args.dataset_name + '-Complete/train'):
            os.makedirs(args.dataset_name + '-Complete/train')
        if not os.path.exists(args.dataset_name + '-Complete/test'):
            os.makedirs(args.dataset_name + '-Complete/test')
        all_data_list = complete_conversations.keys()
        fold_size = int(len(all_data_list)/args.n_folds)
        test_set = random.sample(all_data_list, fold_size)
        train_set = [d for d in all_data_list if d not in test_set]
        for f in train_set:
            with open(args.dataset_name + '-Complete/train/' + str(f), 'w') as output:
                output.write('\n'.join(complete_conversations[f]))
        for f in test_set:
            with open(args.dataset_name + '-Complete/test/' + str(f), 'w') as output:
                output.write('\n'.join(complete_conversations[f]))
    
    elif args.dataset_name == 'Opendialkg':
        max_data_size = 10000
        total_complete_size = 0
        with open('opendialkg.csv.txt') as f:
            reader = csv.reader(f, delimiter = ',', quotechar='"')
            for i, row in enumerate(reader):
                if row[0][0] != '[':
                    continue
                utterances = json.loads(row[0])
                conversation = []
                sender = ''
                for u in utterances:
                    try:
                        if u['sender'] != sender:
                            conversation.append(u['message'])
                        else:
                            conversation[-1] += u['message']
                    except:
                        pass
                if len(conversation) >=4 and len(conversation) <= 10 and len(conversation) % 2 == 0:
                    total_complete_size += 1
                    if total_complete_size <= max_data_size:
                        complete_conversations[i] = conversation
                    else:
                        incomplete_conversations[i] = conversation
                else:
                    incomplete_conversations[i] = conversation
    
        # write to file
        if not os.path.exists(args.dataset_name + '-Complete/train'):
            os.makedirs(args.dataset_name + '-Complete/train')
        if not os.path.exists(args.dataset_name + '-Complete/test'):
            os.makedirs(args.dataset_name + '-Complete/test')
        all_data_list = complete_conversations.keys()
        fold_size = int(len(all_data_list)/args.n_folds)
        test_set = random.sample(all_data_list, fold_size)
        train_set = [d for d in all_data_list if d not in test_set]
        for f in train_set:
            with open(args.dataset_name + '-Complete/train/' + str(f), 'w') as output:
                output.write('\n'.join(complete_conversations[f]))
        for f in test_set:
            with open(args.dataset_name + '-Complete/test/' + str(f), 'w') as output:
                output.write('\n'.join(complete_conversations[f]))

    
    # generate data for reranker fine-tuning
    with open(args.dataset_name + '-parlai-question','w') as f:
        for k in complete_conversations.keys():
            data = complete_conversations[k]
            for id, sentence in enumerate(data):
                if id % 2 == 0 and id != (len(data)-2):
                    f.write('text:')
                    f.write(data[id].strip())
                    f.write('\t')
                elif id % 2 == 1 and id != (len(data) -1):
                    f.write('labels:')
                    f.write(data[id].strip())
                    if id == len(data) - 3 :
                        f.write('\t')
                        f.write('episode_done:True')
                    f.write('\n')
        for k in incomplete_conversations.keys():
            data = incomplete_conversations[k]
            for id, sentence in enumerate(data):
                if id % 2 == 0:
                    f.write('text:')
                    f.write(data[id].strip())
                    f.write('\t')
                elif id % 2 == 1:
                    f.write('labels:')
                    f.write(data[id].strip())
                    f.write('\t')
                    f.write('episode_done:True')
                    f.write('\n')

    with open(args.dataset_name + '-parlai-answer','w') as f:
        for k in complete_conversations.keys():
            data = complete_conversations[k]
            for id, sentence in enumerate(data):
                if id % 2 == 0:
                    f.write('text:')
                    f.write(data[id].strip())
                    f.write('\t')
                elif id % 2 == 1:
                    f.write('labels:')
                    f.write(data[id].strip())
                    if id == len(data) - 1 :
                        f.write('\t')
                        f.write('episode_done:True')
                    f.write('\n')
    
