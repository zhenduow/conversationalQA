import json
import csv
import re
import numpy as np
from copy import deepcopy
import resource
product_name = set(['Windows_7', 'Skype_Android', 'Apps_Windows_10', 'Excel', 'Word',
    'Outlook_Contacts', 'Office_Insider', 'Outlook_Calendar', 'Windows_Insider_Apps',
    'Bing', 'Bing_Maps', 'Windows_RT_8.1', 'Office_Install', 'Office_Account',
    'Games_Windows_10', 'Outlook_Email', 'Outlook_Preview', 'Outlook', 'Skype_Windows_Desktop',
    'PowerPoint', 'Windows_10', 'Skype_iOS', 'Windows_Insider_Games', 'Skype_Linux',
    'Windows_Insider_IE_Edge', 'Windows_Insider_Register', 'Skype_Mac', 'Skype_Web',
    'Skype_Lite', 'Windows_Insider_Preview', 'Windows_Insider_Office', 'Bing_Apps',
    'Skype_Xbox', 'Bing_Ads', 'Bing_Search', 'Windows_8.1', 'Bing_Safety', 'Skype_Windows_10', 'Windows_Mobile'])

max_len = 512

def processing(input_text, max_len):
    '''
    A simple processing function to preprocess the data.
    '''
    result = re.sub('\s\s+', ' ', input_text) # replace more than one whitespace with one.
    result = re.sub(r'http\S+', '', result) # remove URL
    result = ' '.join((result.split()[:max_len]))
    return result.strip()

if __name__ == "__main__":
    with open('MSDialog-Complete.json') as f:
        data = json.load(f)

        for k in data.keys():
            #if data[k]['category'] not in product_name:
            #    continue
            is_answer_label = [1 if ut['is_answer'] == 1 else 0 for ut in data[k]['utterances']]
            has_answer = np.sum(is_answer_label)
            if has_answer == 0:
                # conversation has no answer, so take all user-agent turn pairs as query-cq pairs.
                conversations = []
                for utterance_id, utterance in enumerate(data[k]['utterances']):
                    unprocessed = utterance['utterance']
                    processed = processing(unprocessed  , max_len).strip()
                    if conversations== []:
                        conversations = [data[k]['title'] + '. [SEP] ' + processed]
                        #conversations = [processed]
                    elif utterance['actor_type'] == role:
                        # concatenate all consecutive utterances from one role together
                        conversations[-1] += '. '
                        conversations[-1] += processed
                    else:
                        if processed != '':
                            conversations.append(processed)
                    
                    role = utterance['actor_type']
                    
                if len(conversations) % 2 == 1:
                    conversations = conversations[:-1]
                if 2 <= len(conversations) <= 20 :
                    out = open('MSDialog-Incomplete/' + k, 'w')
                    out.write('\n'.join(conversations))
                    out.close()   
                continue
                
            # conversation has an answer. create query-cq and query-answer pairs.
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
                    if processed != '':
                        conversations.append(processed)
                        
                # adding any agent response to response pool and update it as the final_answer
                if utterance['is_answer'] == 1:
                    final_answer = processed
                    break

                role = utterance['actor_type']
            

            if len(conversations) <= 2 or len(conversations) >= 10 or len(conversations) % 2 or len(conversations[-1].split()) > 512 :
                continue
            for convid in range(len(conversations)):
                conversations[convid] = processing(conversations[convid], max_len)

            out = open('MSDialog-Answer/' + k, 'w')
            out.write('\n'.join(conversations))
            out.close()
