import json
import csv
import re
import numpy as np

response_data = []
answer_data = []
question_data = []
question_id = []

max_len = 360

def processing(input_text, max_len):
    '''
    A simple processing function to preprocess the data.
    '''
    result = re.sub('\s\s+', ' ', input_text) # replace more than one whitespace with one.
    result = re.sub(r'http\S+', '', result) # remove URL
    result = result[:max_len]
    return result.strip()

with open('MSDialog-Complete.json') as f:
    data = json.load(f)

    for k in data.keys():
        final_answer = ''
        is_answer_label = [1 if ut['is_answer'] == 1 else 0 for ut in data[k]['utterances']]
        has_answer = np.sum(is_answer_label)
        if has_answer == 0:
            continue
        conversations = []
        for utterance_id, utterance in enumerate(data[k]['utterances']):
            unprocessed = utterance['utterance']
            processed = processing(unprocessed, max_len).strip()
            if conversations== []:
                conversations = [data[k]['title'] + '. [SEP] ' + processed]
                #conversations = [processed]
            elif utterance['actor_type'] == role:
                # concatenate all consecutive utterances from one role together
                conversations[-1] += '. '
                conversations[-1] += processed
            else:
                if utterance['actor_type'] == 'User' and utterance_id == len(data[k]['utterances']) -1 :
                    break
                conversations.append(processed)
                    
            # adding any agent response to response pool and update it as the final_answer
            if utterance['is_answer'] == 1:
                final_answer = processed
                break

            role = utterance['actor_type']
        
        if len(conversations) <= 2:
            continue

        for final_ut_id in range(len(conversations)):
            if final_ut_id % 2:
                if final_ut_id != len(conversations) -1:
                    response_data.append([k, conversations[final_ut_id], '']) # otherwise append data[k]['title]
            else:
                if final_ut_id == 0:
                    question_data.append(conversations[final_ut_id])
                else:
                    question_data.append(' [SEP] '.join(conversations[:final_ut_id + 1]))
                question_id.append(k)
                    
        if final_answer != '':
            answer_data.append([k, final_answer, '']) 

# check response pool and answer pool are disjoint. It does happen because Microsoft agents use template answers very occasionally.
#print(set.intersection(set([i[1] for i in response_data]),set([i[1] for i in answer_data])))

question_response_pair = []
question_answer_pair = []
for i in range(len(question_data)):
    question_response_pair.append([question_data[i], [rd[1] for rd in response_data]])
    question_answer_pair.append([question_data[i], [ad[1] for ad in answer_data]])

with open('DPR_response.tsv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(response_data)):
        writer.writerow(response_data[i])

with open('DPR_answer.tsv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(answer_data)):
        writer.writerow(answer_data[i])


with open('DPR_response_test.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(question_response_pair)):
        writer.writerow(question_response_pair[i])

with open('DPR_answer_test.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(question_answer_pair)):
        writer.writerow(question_answer_pair[i])

with open('DPR_gold','w') as f:
    for i in range(len(question_id)):
        f.write(str(question_id[i]))
        f.write('\n')


