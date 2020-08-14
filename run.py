from user import User
from dataset import ConversationDataset
from agent import Agent
from bm25ranker import BM25Ranker
import logging
import numpy as np
import random
import json
import csv
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
train_size = int(train_test_split_ratio * dataset_size)
use_top_k = 1
exp_name = 'product'

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    dataset = ConversationDataset('data/MSDialog-Product/', dataset_size)
    conversation_ids = list(dataset.conversations.keys())
    random.seed(13)
    random.shuffle(conversation_ids)
    train_ids = conversation_ids[:train_size]
    test_ids = conversation_ids[train_size:]
    user = User(dataset, cq_reward = cq_reward, cq_penalty = cq_penalty)
    agent = Agent(lr=0.0001, input_dims = 2* use_top_k * observation_dim, n_actions=action_num, gamma = agent_gamma)
    memory = {}

    parser = argparse.ArgumentParser()
    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--qa_file', type=str, default='/raid/zhenduow/conversationalQA/data/' + exp_name + '/DPR_qafile',
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--ctx_file', type=str, default='/raid/zhenduow/conversationalQA/data/DPR_question_' + exp_name + '.tsv',
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
    parser.add_argument("--initialize_index", default = True)

    question_retriever_args = parser.parse_args()
    question_retriever_args.model_file = '/raid/zhenduow/conversationalQA/DPR/checkpoint/retriever/multiset/bert-base-encoder.cp'
    setup_args_gpu(question_retriever_args)
    question_retriever_args, question_retriever, question_index_buffer_sz = main(question_retriever_args)

    answer_retriever_args = deepcopy(question_retriever_args)
    answer_retriever_args.ctx_file = '/raid/zhenduow/conversationalQA/data/DPR_answer_' + exp_name + '.tsv'
    answer_retriever_args.encoded_ctx_file = '/raid/zhenduow/conversationalQA/data/answer_embeddings_' + exp_name + '_0'
    answer_retriever_args.out_file = '/raid/zhenduow/conversationalQA/data/' + exp_name + '/answer_retrieval'
    answer_retriever_args, answer_retriever, answer_index_buffer_sz = main(answer_retriever_args)


    for i in range(train_iter):
        train_scores = []
        train_q0_scores = []
        train_q1_scores = []
        train_q2_scores = []

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
                        riwriter = csv.writer(retrieverinput, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        riwriter.writerow([obs, ['']])

                    retrieve(question_retriever_args, question_retriever, question_index_buffer_sz)
                    question_retriever_args.initialize_index = False
                    retrieve(answer_retriever_args, answer_retriever, answer_index_buffer_sz)
                    answer_retriever_args.initialize_index = False

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
                            riwriter = csv.writer(retrieverinput, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            riwriter.writerow([obs, ['']])

                        retrieve(question_retriever_args, question_retriever, question_index_buffer_sz)
                        retrieve(answer_retriever_args, answer_retriever, answer_index_buffer_sz)

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

                agent.joint_learn((obs, question, answer), answer_reward, question_reward, (obs_, question_, answer_))
                obs = obs_



        # test the agent
        for test_id in test_ids:
            obs = user.initialize_state(test_id)
            ignore_questions = []
            q_done = False

            while not q_done:
                if obs in memory.keys():
                    question, answer = memory[obs][0], memory[obs][1]
                
                else:
                    # get retriever results
                    with open(question_retriever_args.qa_file, 'w') as retrieverinput:
                        retrieverinput.write(obs)

                    retrieve(question_retriever_args, question_retriever, question_index_buffer_sz)
                    retrieve(answer_retriever_args, answer_retriever, answer_index_buffer_sz)

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
                _, answer_reward, _, _ = user.update_state(test_id, obs, 0, question, answer, use_top_k = use_top_k)
                obs_, question_reward, q_done, good_question = user.update_state(test_id, obs, 1, question, answer, use_top_k = use_top_k)
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

                obs = obs_
