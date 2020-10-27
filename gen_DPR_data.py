import json
import csv
import re
import numpy as np
from copy import deepcopy
import resource
import glob
import subprocess
import random

def limit_memory(maxsize): 
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard)) 

response_data = []
answer_data = []
question_data = []
DPR_answer_training = []
max_len = 512
all_orders = []

def processing(input_text, max_len):
    '''
    A simple processing function to preprocess the data.
    '''
    result = re.sub('\s\s+', ' ', input_text) # replace more than one whitespace with one.
    result = re.sub(r'http\S+', '', result) # remove URL
    result = result[:max_len]
    return result.strip()


def generate_orders(current_list, res):
    if res == []:
        all_orders.append(current_list)
    else:
        for r in range(len(res)):
            current_list_copy = deepcopy(current_list)
            current_list_copy.append(res[r])
            generate_orders(current_list_copy, res[r+1:])
            

def generate_all_query(user_ut, agent_ut):
    assert len(user_ut) == len(agent_ut)
    ids = list(range(len(user_ut))[:-1])
    all_query = [{'question':user_ut[0],'positive_ctxs':[{'title': '', 'text': agent_ut[0]}]}]
    all_orders[:] = []
    generate_orders([], ids)
    for order in all_orders:
        query = user_ut[0]
        for o in order:
            query += ' [SEP] ' + agent_ut[o] + ' [SEP] ' + user_ut[o+1]
        try:
            all_query.append({'question': query, 'positive_ctxs': [{'title': '', 'text': agent_ut[order[-1] + 1]}]})
        except:
            pass
    return all_query

question_data = []
limit_memory(1e11)
all_data_list = glob.glob('MSDialog-Answer/train/*')
for data_file in all_data_list:
    f = open(data_file)
    data = f.readlines()
    data_id = data_file.split('/')[1]
    agent_ut = []
    user_ut = []
    final_query = data[0]
    for line_num in range(len(data)):
        if line_num % 2:
            agent_ut.append(data[line_num].strip())
            if line_num != len(data) - 1:
                response_data.append([data_id, data[line_num].strip(), ''])
            final_query += ' [SEP] ' + data[line_num].strip()
        else:
            user_ut.append(data[line_num].strip())
            if line_num > 0:
                final_query += ' [SEP] ' + data[line_num].strip()
    
    # creating training set for question retriever DPR
    
    negative_conversation = random.choice(all_data_list)
    try:
        assert negative_conversation != data_file
    except:
        negative_conversation = random.choice(all_data_list)
    neg_f = open(negative_conversation)
    neg_data = neg_f.readlines()
    neg_linenum = [lineid for lineid in range(len(neg_data)) if lineid % 2]
    all_query = generate_all_query(user_ut, agent_ut)
    for idx, item in enumerate(all_query):
        all_query[idx]['negative_ctxs'] = [neg_data[random.choice(neg_linenum)]]
        all_query[idx]['hard_negative_ctxs'] = [neg_data[random.choice(neg_linenum)]]
    question_data.extend(all_query)

    answer_data.append([data_id, data[-1], '']) 

    DPR_answer_training.append({'question': final_query, 'positive_ctxs': [{'title': '', 'text': agent_ut[-1]}], 'negative_ctxs': [neg_data[-1]], 'hard_negative_ctxs': [neg_data[-1]]}) 



all_data_list = glob.glob('MSDialog-Incomplete/*')
for data_file in all_data_list:
    f = open(data_file)
    data = f.readlines()
    data_id = data_file.split('/')[1]
    agent_ut = []
    user_ut = []
    final_query = data[0]
    for line_num in range(len(data)):
        if line_num % 2:
            agent_ut.append(data[line_num].strip())
            final_query += ' [SEP] ' + data[line_num].strip()
        else:
            user_ut.append(data[line_num].strip())
            if line_num > 0:
                final_query += ' [SEP] ' + data[line_num].strip()
    
    # creating training set for question retriever DPR
    
    negative_conversation = random.choice(all_data_list)
    try:
        assert negative_conversation != data_file
    except:
        negative_conversation = random.choice(all_data_list)
    neg_f = open(negative_conversation)
    neg_data = neg_f.readlines()
    neg_linenum = [lineid for lineid in range(len(neg_data)) if lineid % 2]
    all_query = generate_all_query(user_ut, agent_ut)
    for idx, item in enumerate(all_query):
        all_query[idx]['negative_ctxs'] = [neg_data[random.choice(neg_linenum)]]
        all_query[idx]['hard_negative_ctxs'] = [neg_data[random.choice(neg_linenum)]]
    question_data.extend(all_query)



with open('all_questions_train.tsv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(response_data)):
        writer.writerow(response_data[i])

with open('all_answers_train.tsv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(answer_data)):
        writer.writerow(answer_data[i])

with open('DPR_question_train','w', newline='') as jsonfile:
    json.dump(question_data, jsonfile)

with open('DPR_answer_train','w', newline='') as jsonfile:
    json.dump(DPR_answer_training, jsonfile)

'''
subprocess.check_call(["/usr/bin/python3.6", '/raid/zhenduow/conversationalQA/DPR/generate_dense_embeddings.py',\
        "--model_file", "/raid/zhenduow/conversationalQA/DPR/checkpoint/retriever/multiset/dpr_biencoder.question",\
        "--ctx_file", "/raid/zhenduow/conversationalQA/data/all_questions_train.tsv",\
        "--out_file", "/raid/zhenduow/conversationalQA/data/all_questions_train_embeddings"])

subprocess.check_call(["/usr/bin/python3.6", '/raid/zhenduow/conversationalQA/DPR/generate_dense_embeddings.py',\
        "--model_file", "/raid/zhenduow/conversationalQA/DPR/checkpoint/retriever/multiset/dpr_biencoder.answer",\
        "--ctx_file", "/raid/zhenduow/conversationalQA/data/all_answers_train.tsv",\
        "--out_file", "/raid/zhenduow/conversationalQA/data/all_answers_train_embeddings"])
'''
