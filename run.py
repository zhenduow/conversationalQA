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
from DPR import *
from DPR.dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, add_tokenizer_params, add_cuda_params
from DPR.dense_retriever import main, retrieve
observation_dim = 768
action_num = 2
cq_reward = 0.49
cq_penalty = -0.5
agent_gamma = 0.25
pretrain_iter = 3
train_iter = 10
dataset_size = 1496
train_test_split_ratio = 0.8
train_size = train_test_split_ratio * dataset_size
use_top_k = 1
exp_name = 'final_product'

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    dataset = ConversationDataset('data/MSDialog-Product/.json', dataset_size)
    conversation_ids = list(dataset.conversations.keys())
    random.seed(13)
    random.shuffle(conversation_ids)
    train_ids = conversation_ids[:train_size]
    test_ids = conversation_ids[train_size:]
    user = User(dataset, 2, 5)
    agent = Agent(lr=0.0001, input_dims=observation_dim, n_actions=action_num, gamma = agent_gamma)
    memory = {}

    parser = argparse.ArgumentParser()
    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--qa_file', required=True, type=str, default='/raid/zhenduow/conversationalQA/data/' + exp_name + '/DPR_qafile',
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--ctx_file', required=True, type=str, default='/raid/zhenduow/conversationalQA/data/DPR_response_' + exp_name + '.csv',
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    parser.add_argument('--encoded_ctx_file', type=str, default='/raid/zhenduow/conversationalQA/data/question_embeddings_' + exp_name + '_0',
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--out_file', type=str, default='/raid/zhenduow/conversationalQA/data/' + exp_name + '/question_retrieval',
                        help='output .tsv file path to write results to ')
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'])
    parser.add_argument('--n-docs', type=int, default=10, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,)
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')

    question_retriever_args = parser.parse_args()
    question_retriever_args.model_file = '/raid/zhenduow/conversationalQA/DPR/checkpoint/retriever/multiset/bert-base-encoder.cp'
    setup_args_gpu(question_retriever_args)
    question_retriever_args, retriever, index_buffer_sz = main(question_retriever_args)

    answer_retriever_args = deepcopy(question_retriever_args)
    answer_retriever_args.ctx_file = '/raid/zhenduow/conversationalQA/data/DPR_answer_' + exp_name + '.csv'
    answer_retriever_args.encoded_ctx_file = '/raid/zhenduow/conversationalQA/data/answer_embeddings_' + exp_name + '_0'
    answer_retriever_args.out_file = '/raid/zhenduow/conversationalQA/data/' + exp_name + '/answer_retrieval'


    for i in range(train_iter):
        for conv_id in train_ids:
            obs = user.initialize_state(conv_id)
            ignore_questions = []
            q_done = False

            while not q_done:
                if obs in memory.keys():
                    question, answer = memory[obs][0], memory[obs][1]
                
                else:
                    # get retriever results
                    with open(question_retriever_args.qa_file, 'w') as retrieverinput:
                        retrieverinput.write(obs)

                    retrieve(question_retriever_args, retriever, index_buffer_sz)
                    retrieve(answer_retriever_args, retriever, index_buffer_sz)

                    with open(question_retriever_args.out_file) as question_f:
                        jsondata = json.load(question_f)
                        retrieved_question = [ctx['text'] for ctx in jsondata[0]['ctxs']]
                    with open(answer_retriever_args.out_file) as answer_f:
                        jsondata = json.load(answer_f)
                        retrieved_answer = [ctx['text'] for ctx in jsondata[0]['ctxs']]

                    retrieved_question = [rq for rq in retrieved_question if rq not in ignore_questions]

                    with open('retrieved_question', 'w') as qrerankerinput:
                        for rq in retrieved_question:
                            qrerankerinput.write(rq)
                            qrerankerinput.write('\n')
                    with open('retrieved_answer', 'w') as arerankerinput:
                        for ra in retrieved_answer:
                            arerankerinput.write(ra)
                            arerankerinput.write('\n')

                    # get reranker results
                    question = Interactive.main(model = 'transformer/polyencoder', \
                        model_file = 'zoo:pretrained_transformers/model_poly/msdialogquestion',  \
                        encode_candidate_vecs = 'true', eval_candidates ='fixed',  \
                        fixed_candidates_path = 'retrieved_question', \
                        human_input=obs, fixed_candidate_vecs = 'replace')
                    answer = Interactive.main(model = 'transformer/polyencoder', \
                        model_file = 'zoo:pretrained_transformers/model_poly/msdialogmodel',  \
                        encode_candidate_vecs = 'true', eval_candidates ='fixed',  \
                        fixed_candidates_path = 'retrieved_answer', \
                        human_input=obs,  fixed_candidate_vecs = 'replace')
                    
                    memory[obs] = [question, answer]

                action = agent.choose_action(obs, question, answer)
                _, answer_reward, _, _ = user.update_state(conv_id, obs, 0, question, answer, use_top_k = use_top_k)
                obs_, question_reward, q_done, good_question = user.update_state(conv_id, obs, 1, question, answer, use_top_k = use_top_k)
                action_reward = [answer_reward, question_reward][action]
                print('answer reward', answer_reward, 'question reward', question_reward, 'q done', q_done)

                if not q_done:
                    ignore_questions.append(good_question)
                    if obs_ in memory.keys():
                        question_, answer_ = memory[obs_][0], memory[obs_][1]
                    else:
                        # get retriever results
                        with open(question_retriever_args.qa_file, 'w') as retrieverinput:
                            retrieverinput.write(obs_)

                        retrieve(question_retriever_args, retriever, index_buffer_sz)
                        retrieve(answer_retriever_args, retriever, index_buffer_sz)

                        with open(answer_retriever_args.out_file) as question_f:
                            jsondata = json.load(question_f)
                            retrieved_question = [ctx['text'] for ctx in jsondata[0]['ctxs']]
                        with open(answer_retriever_args.out_file) as answer_f:
                            jsondata = json.load(answer_f)
                            retrieved_answer = [ctx['text'] for ctx in jsondata[0]['ctxs']]

                        retrieved_question = [rq for rq in retrieved_question if rq not in ignore_questions]

                        with open('retrieved_question', 'w') as qrerankerinput:
                            for rq in retrieved_question:
                                qrerankerinput.write(rq)
                                qrerankerinput.write('\n')
                        with open('retrieved_answer', 'w') as arerankerinput:
                            for ra in retrieved_answer:
                                arerankerinput.write(ra)
                                arerankerinput.write('\n')

                        # get reranker results
                        question_ = Interactive.main(model = 'transformer/polyencoder', \
                            model_file = 'zoo:pretrained_transformers/model_poly/msdialogquestion',  \
                            encode_candidate_vecs = 'true', eval_candidates ='fixed',  \
                            fixed_candidates_path = 'retrieved_question', \
                            human_input=obs_, fixed_candidate_vecs = 'replace')
                        answer_ = Interactive.main(model = 'transformer/polyencoder', \
                            model_file = 'zoo:pretrained_transformers/model_poly/msdialogmodel',  \
                            encode_candidate_vecs = 'true', eval_candidates ='fixed',  \
                            fixed_candidates_path = 'retrieved_answer', \
                            human_input=obs_,  fixed_candidate_vecs = 'replace')
                        
                        memory[obs_] = [question_, answer_]

                else:
                    question_, answer_ = None, None

                agent.joint_learn((obs, question, answer)), answer_reward, question_reward, (obs_, question_, answer_)):
                obs = obs_



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
