from user import User
from dataset import ConversationDataset
from agent import Agent
from bm25ranker import BM25Ranker
import logging
import numpy as np
import random
import json
from parlai.scripts.interactive import Interactive
from copy import deepcopy
import argparse
#from DPR import *
#from DPR.dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, add_tokenizer_params, add_cuda_params
#from DPR.dense_retriever import main
observation_dim = 768
action_num = 2
max_iter = 5000
eval_batch_size = 100
test_size = 1000

#if __name__ == "__main__":
#   logging.getLogger().setLevel(logging.INFO)
#    dataset = ConversationDataset('./data/MSDialog-Intent.json')
#    user = User(dataset, 2, 5)
#    user.simulate('96')

def set_up_retriever_args():
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--qa_file', required=True, type=str, default='data/DPR_test.csv',
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--ctx_file', required=True, type=str, default='data/DPR_response.tsv',
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    parser.add_argument('--encoded_ctx_file', type=str, default='DPR/question_candidates_vector_0',
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--out_file', type=str, default='responser_result',
                        help='output .tsv file path to write results to ')
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'],
                        help="Answer matching logic type")
    parser.add_argument('--n-docs', type=int, default=200, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')
    parser.add_argument("--model_file", type=str, default='DPR/checkpoint/retriever/multiset/bert-base-encoder.cp')

    args = parser.parse_args()
    setup_args_gpu(args)
    return args


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    dataset = ConversationDataset('./data/MSDialog-Intent.json')
    n_conversations = len(dataset.conversations.keys())
    random.seed(13)
    user = User(dataset, 2, 5)

    bm25agent = BM25Ranker(dataset)

    scores = []
    eps_history = []
    worse_choice = []
    answer_only_scores = []
    question_only_scores = []

    agent = Agent(lr=0.0001, input_dims=observation_dim,
                  n_actions=action_num)

    '''
    responser_args = set_up_responser_args()
    answerer_args = deepcopy(responser_args)
    answerer_args.ctx_file = 'data\DPR_answer.tsv'
    answerer_args.encoded_ctx_file = 'DPR/answer_candidates_vector_0'
    answerer_args.out_file = 'answerer_result'
    '''
    
    responser_retrieval = {}
    answerer_retrieval = {}
    with open('data/responser_output') as f:
         data = json.load(f)
         for item in data:
             responser_retrieval[item['question']] = [response['text'] for response in item['ctxs']]
    with open('data/answerer_output') as f:
         data = json.load(f)
         for item in data:
             answerer_retrieval[item['question']] = [response['text'] for response in item['ctxs']]

    #print(responser_retrieval.keys())

    for i in range(max_iter):
        score = 0
        done = False
        conversation_id = random.choice(list(dataset.conversations.keys()))
        obs = user.initialize_state(conversation_id)
        while not done:
            try: # get retriever results
                retrieved_question = responser_retrieval[obs]
                retrieved_answer = answerer_retrieval[obs]
                with open('retrieved_question','w') as f:
                    for q in retrieved_question:
                        f.writelines(q+'\n')
                    
                with open('retrieved_answer','w') as f:
                    for q in retrieved_answer:
                        f.writelines(q+'\n')
            except:
                print('Retriever failed.')
                with open('retrieved_question','w') as f:
                    for q in range(10):
                        f.writelines('\n')
                    
                with open('retrieved_answer','w') as f:
                    for q in range(10):
                        f.writelines('\n')

            # get question and answer reranker results
            question = Interactive.main(model = 'transformer/polyencoder', model_file = 'zoo:pretrained_transformers/model_poly/model',  encode_candidate_vecs = 'true', eval_candidates ='fixed',  fixed_candidates_path = 'retrieved_question', human_input=obs, fixed_candidate_vecs = 'replace')
            answer = Interactive.main(model = 'transformer/polyencoder', model_file = 'zoo:pretrained_transformers/model_poly/model',  encode_candidate_vecs = 'true', eval_candidates ='fixed',  fixed_candidates_path = 'retrieved_answer', human_input=obs,  fixed_candidate_vecs = 'replace')
            full_obs = obs + '  [SEP] ' + question[0] + ' [SEP] ' + answer[0] # full observation is the contatenation of the conversation, the best retrieved question, and the best retrieved answer
            action = agent.choose_action(full_obs)
            obs_, reward, done = user.update_state(conversation_id, obs, action, question, answer)
            print('Action: ' + str(action) + ' reward: ' + str(reward) + ' Done: ' +str(done))

            if reward == 0:
                # evaluate the other action choice
                _, other_reward, _ = user.update_state(conversation_id, obs, 1-action, question, answer)
                print(question) if action else print(answer)
                print("Other reward:", other_reward)
                if other_reward > reward:
                    worse_choice.append(1)
                else:
                    worse_choice.append(0)

            # compute the next state for learning
            full_obs_ = full_obs
            if not done:
                question_ = Interactive.main(model = 'transformer/polyencoder', model_file = 'zoo:pretrained_transformers/model_poly/model',  encode_candidate_vecs = 'true', eval_candidates ='fixed',  fixed_candidates_path = 'retrieved_question', human_input=obs_, fixed_candidate_vecs = 'replace')
                answer_ = Interactive.main(model = 'transformer/polyencoder', model_file = 'zoo:pretrained_transformers/model_poly/model',  encode_candidate_vecs = 'true', eval_candidates ='fixed',  fixed_candidates_path = 'retrieved_answer', human_input=obs_, fixed_candidate_vecs = 'replace')
                full_obs_ = obs_ + '  [SEP] ' + question_[0] + ' [SEP] ' + answer_[0]
            
            # learn the transition
            #agent.learn(full_obs, action, reward, full_obs_)

            # update
            score += reward
            # get answer only score
            _, answer_reward, _ = user.update_state(conversation_id, obs, 0, question, answer)
            # get question only score
            _, question_reward, _ = user.update_state(conversation_id, obs, 1, question, answer)
            answer_only_scores.append(answer_reward)
            question_only_scores.append(question_reward)
            obs = obs_
            
            # learn the transition
            if question_reward != 0 or answer_reward != 0:
                agent.learn(full_obs, action, reward, full_obs_)

        scores.append(score)
        eps_history.append(agent.epsilon)
        
        if i % eval_batch_size == 0:
            avg_score = np.mean(scores[-eval_batch_size:])
            worse_choice_made = np.sum(worse_choice[-eval_batch_size:])
            print('episode ', i, 'avg score %.6f worse_choice_made %.0f epsilon %.2f' %
                  (avg_score, worse_choice_made, agent.epsilon))
            print("Answer only:", np.mean(answer_only_scores[-eval_batch_size:]))
            print("Answer acc:", np.sum([1 if a == 1 else 0 for a in answer_only_scores][-eval_batch_size:]))
            print("Question only:", np.mean(question_only_scores[-eval_batch_size:]))
            print("Question acc:", np.sum([1 if a == 0.01 else 0 for a in question_only_scores][-eval_batch_size:]))



    # test the agent
    test_scores = []
    test_id_list = random.sample(list(dataset.conversations.keys()), test_size)
    for i in test_id_list:
        done = False
        test_conversation_id = random.choice(list(dataset.conversations.keys()))
        test_obs = user.initialize_state(test_conversation_id)

        while not done:
            try: # get retriever results
                retrieved_question = responser_retrieval[test_obs]
                retrieved_answer = answerer_retrieval[test_obs]
                with open('retrieved_question','w') as f:
                    for q in retrieved_question:
                        f.writelines(q+'\n')
                    
                with open('retrieved_answer','w') as f:
                    for q in retrieved_answer:
                        f.writelines(q+'\n')
            except:
                print('Retriever failed.')
                with open('retrieved_question','w') as f:
                    for q in range(10):
                        f.writelines('\n')
                    
                with open('retrieved_answer','w') as f:
                    for q in range(10):
                        f.writelines('\n')


            question = Interactive.main(model = 'transformer/polyencoder', model_file = 'zoo:pretrained_transformers/model_poly/model',  encode_candidate_vecs = 'true', eval_candidates ='fixed',  fixed_candidates_path = 'retrieved_question', human_input=test_obs, fixed_candidate_vecs = 'replace')
            answer = Interactive.main(model = 'transformer/polyencoder', model_file = 'zoo:pretrained_transformers/model_poly/model',  encode_candidate_vecs = 'true', eval_candidates ='fixed',  fixed_candidates_path = 'retrieved_answer', human_input=test_obs,  fixed_candidate_vecs = 'replace')
            full_test_obs = test_obs + '  [SEP] ' + question[0] + ' [SEP] ' + answer[0] # full observation is the contatenation of the conversation, the best retrieved question, and the best retrieved answer
            action = agent.choose_action(full_test_obs)
            test_obs_, reward, done = user.update_state(test_conversation_id, test_obs, action, question, answer)          
            score += reward
  
            test_obs = test_obs_
            test_scores.append(score)
        print("Test score %.6f" % np.mean(test_scores))
