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
question_id = []
max_len = 360
all_orders = []
dataset_size = 1496
train_size = int(0.8 * dataset_size)

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
        all_query.append({'question': query, 'positive_ctxs': [{'title': '', 'text': agent_ut[order[-1] + 1]}]})
    return all_query

question_data = []
limit_memory(1e11)
all_data_list = glob.glob('MSDialog-Product/*')
total = 0
for data_file in all_data_list:
    total += 1
    if total > dataset_size:
        break
    f = open(data_file)
    data = f.readlines()
    data_id = data_file.split('/')[1]
    agent_ut = []
    user_ut = []
    for line_num in range(len(data)):
        if line_num % 2:
            agent_ut.append(data[line_num].strip())
            if line_num != len(data) - 1:
                response_data.append([data_id, data[line_num].strip(), ''])
        else:
            user_ut.append(data[line_num].strip())
    
    # creating training set for question retriever DPR
    if total <= train_size:
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
        all_query_id = [data_id] * len(all_query)
        question_data.extend(all_query)
        question_id.extend(all_query_id)
    

    answer_data.append([data_id, data[-1], '']) 

'''
question_response_pair = []
question_answer_pair = []
for i in range(len(question_data)):
    question_response_pair.append([question_data[i], [rd[1] for rd in response_data]])
    question_answer_pair.append([question_data[i], [ad[1] for ad in answer_data]])
'''


with open('DPR_response_product.tsv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(response_data)):
        writer.writerow(response_data[i])

with open('DPR_answer_product.tsv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(answer_data)):
        writer.writerow(answer_data[i])
'''
with open('DPR_gold','w') as f:
    for i in range(len(question_id)):
        f.write(str(question_id[i]))
        f.write('\n')
'''

with open('DPR_question_retriever_training','w', newline='') as jsonfile:
    json.dump(question_data, jsonfile)

'''
with open('DPR_answer_test.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(question_answer_pair)):
        writer.writerow(question_answer_pair[i])
'''


subprocess.check_call(["/usr/bin/python3.6", '/raid/zhenduow/conversationalQA/DPR/generate_dense_embeddings.py',\
        "--model_file", "/raid/zhenduow/conversationalQA/DPR/checkpoint/retriever/multiset/bert-base-encoder.cp",\
        "--ctx_file", "/raid/zhenduow/conversationalQA/data/DPR_response_product.tsv",\
        "--out_file", "/raid/zhenduow/conversationalQA/data/response_embeddings_product"])

subprocess.check_call(["/usr/bin/python3.6", '/raid/zhenduow/conversationalQA/DPR/generate_dense_embeddings.py',\
        "--model_file", "/raid/zhenduow/conversationalQA/DPR/checkpoint/retriever/multiset/bert-base-encoder.cp",\
        "--ctx_file", "/raid/zhenduow/conversationalQA/data/DPR_answer_product.tsv",\
        "--out_file", "/raid/zhenduow/conversationalQA/data/answer_embeddings_product"])
