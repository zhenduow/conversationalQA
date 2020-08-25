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
from DPR.dpr.options import add_encoder_params, setup_args_gpu, set_encoder_params_from_state, add_tokenizer_params, add_cuda_params
from DPR.dense_retriever import main, retrieve
observation_dim = 768
action_num = 2
cq_reward = 0.49
cq_penalty = -0.5
agent_gamma = 0.25
pretrain_iter = 3
train_iter = 20
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
    agent = Agent(lr=1e-4, input_dims = 2* use_top_k * observation_dim, n_actions=action_num, gamma = agent_gamma)
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
        train_scores, train_q0_scores, train_q1_scores, train_q2_scores, train_oracle_scores = [],[],[],[],[]
        train_worse, train_q0_worse, train_q1_worse, train_q2_worse, train_oracle_worse = [],[],[],[],[]
        train_correct, train_q0_correct, train_q1_correct, train_q2_correct, train_oracle_correct = [],[],[],[],[]
        for conv_serial, conv_id in enumerate(train_ids):
            obs = user.initialize_state(conv_id)
            ignore_questions = []
            n_round = 0
            q_done = False
            stop = False
            print('-------- train conversation %.0f/%.0f --------' % (conv_serial + 1, len(train_ids)))
            while not q_done:
                print('-------- round %.0f --------' % (n_round))
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

                    with open('/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_question', 'w') as qrerankerinput:
                        for rq in retrieved_question:
                            qrerankerinput.write(rq)
                            qrerankerinput.write('\n')
                    with open('/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_answer', 'w') as arerankerinput:
                        for ra in retrieved_answer:
                            arerankerinput.write(ra)
                            arerankerinput.write('\n')

                    # get reranker results
                    question = Interactive.main(model = 'transformer/polyencoder', \
                        model_file = 'zoo:pretrained_transformers/model_poly/msdialogquestion',  \
                        encode_candidate_vecs = 'true', eval_candidates ='fixed',  \
                        fixed_candidates_path = '/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_question', \
                            force_fp16_tokens = True,\
                        human_input=obs, fixed_candidate_vecs = 'replace')
                    answer = Interactive.main(model = 'transformer/polyencoder', \
                        model_file = 'zoo:pretrained_transformers/model_poly/msdialoganswer',  \
                        encode_candidate_vecs = 'true', eval_candidates ='fixed',  \
                        fixed_candidates_path = '/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_answer', \
                        human_input=obs,  fixed_candidate_vecs = 'replace')
                    
                    memory[obs] = [question, answer]

                action = agent.choose_action(obs, question, answer)
                obs_, question_reward, q_done, good_question = user.update_state(conv_id, obs, 1, question, answer, use_top_k = use_top_k)
                _, answer_reward, _, _ = user.update_state(conv_id, obs, 0, question, answer, use_top_k = use_top_k)
                action_reward = [answer_reward, question_reward][action]
                print('action', action, 'answer reward', answer_reward, 'question reward', question_reward, 'q done', q_done)

                if not q_done:
                    ignore_questions.append(good_question)
                    if obs_ in memory.keys():
                        question_, answer_ = memory[obs_][0], memory[obs_][1]
                    else:
                        # get retriever results
                        
                        with open(question_retriever_args.qa_file, 'w') as retrieverinput:
                            riwriter = csv.writer(retrieverinput, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            riwriter.writerow([obs_, ['']])

                        retrieve(question_retriever_args, question_retriever, question_index_buffer_sz)
                        retrieve(answer_retriever_args, answer_retriever, answer_index_buffer_sz)

                        with open(answer_retriever_args.out_file) as question_f:
                            jsondata = json.load(question_f)
                            retrieved_question = [ctx['text'] for ctx in jsondata[0]['ctxs']]
                        with open(answer_retriever_args.out_file) as answer_f:
                            jsondata = json.load(answer_f)
                            retrieved_answer = [ctx['text'] for ctx in jsondata[0]['ctxs']]

                        retrieved_question = [rq for rq in retrieved_question if rq not in ignore_questions]

                        with open('/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_question', 'w') as qrerankerinput:
                            for rq in retrieved_question:
                                qrerankerinput.write(rq)
                                qrerankerinput.write('\n')
                        with open('/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_answer', 'w') as arerankerinput:
                            for ra in retrieved_answer:
                                arerankerinput.write(ra)
                                arerankerinput.write('\n')

                        # get reranker results
                        question_ = Interactive.main(model = 'transformer/polyencoder', \
                            model_file = 'zoo:pretrained_transformers/model_poly/msdialogquestion',  \
                            encode_candidate_vecs = 'true', eval_candidates ='fixed',  \
                            fixed_candidates_path = '/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_question', \
                            force_fp16_tokens = True,\
                            human_input=obs_, fixed_candidate_vecs = 'replace')
                        answer_ = Interactive.main(model = 'transformer/polyencoder', \
                            model_file = 'zoo:pretrained_transformers/model_poly/msdialoganswer',  \
                            encode_candidate_vecs = 'true', eval_candidates ='fixed',  \
                            fixed_candidates_path = '/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_answer', \
                            human_input=obs_,  fixed_candidate_vecs = 'replace')
                        
                        memory[obs_] = [question_, answer_]

                else:
                    question_, answer_ = None, None

                agent.joint_learn((obs, question, answer), answer_reward, question_reward, (obs_, question_, answer_))

                # evaluation
                if (action == 0 or (action == 1 and question_reward == cq_penalty)) and not stop:
                    stop = True
                    train_scores.append(answer_reward if action == 0 else 0)
                    if action == 0 and answer_reward == 1.0:
                        train_correct.append(conv_id) 
                    train_worse.append(1 if (action == 0 and answer_reward < float(1/use_top_k) and question_reward == cq_reward) \
                        or (action == 1 and answer_reward > 0 and question_reward == cq_penalty) else 0)
                if n_round == 0:
                    train_q0_scores.append(answer_reward)
                    train_q0_worse.append(1 if answer_reward < float(1/use_top_k) and question_reward == cq_reward else 0)
                    if answer_reward == 1:
                        train_q0_correct.append(conv_id)
                    if q_done:
                        train_q1_scores.append(0)
                        train_q2_scores.append(0)
                        train_q1_worse.append(1)
                        train_q2_worse.append(1)
                elif n_round == 1:
                    train_q1_scores.append(answer_reward)
                    train_q1_worse.append(1 if answer_reward < float(1/use_top_k) and question_reward == cq_reward else 0)
                    if answer_reward == 1:
                        train_q1_correct.append(conv_id)
                    if q_done:
                        train_q2_scores.append(0)
                        train_q2_worse.append(1)
                elif n_round == 2:
                    train_q2_scores.append(answer_reward)
                    train_q2_worse.append(1 if answer_reward < float(1/use_top_k) and question_reward == cq_reward else 0)
                    if answer_reward == 1:
                        train_q2_correct.append(conv_id)

                obs = obs_
                n_round += 1


        assert len(train_scores) == len(train_q0_scores)
        assert len(train_q1_scores) == len(train_q0_scores)
        assert len(train_q1_scores) == len(train_q2_scores)
        for oi in range(len(train_scores)):
            train_oracle_scores.append(max(train_q0_scores[oi], train_q1_scores[oi], train_q2_scores[oi]))
            train_oracle_worse.append(min(train_q0_scores[oi], train_q1_scores[oi], train_q2_scores[oi]))
        train_oracle_correct = list(set(train_correct + train_q0_correct + train_q2_correct))

        print("Train epoch %.0f, acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (i, np.mean([1 if score == 1 else 0 for score in train_scores]), np.mean(train_scores), np.mean(train_worse)))
        print("q0 acc %.0f, avgmrr %.6f, worse decisions %.0f" % 
            (np.mean([1 if score == 1 else 0 for score in train_q0_scores]), np.mean(train_q0_scores), np.mean(train_q0_worse)))
        print("q1 acc %.0f, avgmrr %.6f, worse decisions %.0f" % 
            (np.mean([1 if score == 1 else 0 for score in train_q1_scores]), np.mean(train_q1_scores), np.mean(train_q1_worse)))
        print("q2 acc %.0f, avgmrr %.6f, worse decisions %.0f" % 
            (np.mean([1 if score == 1 else 0 for score in train_q2_scores]), np.mean(train_q2_scores), np.mean(train_q2_worse)))
        print("oracle acc %.0f, avgmrr %.6f, worse decisions %.0f" % 
            (np.mean([1 if score == 1 else 0 for score in train_oracle_scores]), np.mean(train_oracle_scores), np.mean(train_oracle_worse)))
        print(train_correct)
        print(train_q0_correct)
        print(train_q1_correct)
        print(train_q2_correct)
        print(train_oracle_correct)
        print("avg loss", np.mean(agent.loss_history))

        ## test
        test_scores, test_q0_scores, test_q1_scores, test_q2_scores, test_oracle_scores = [],[],[],[],[]
        test_worse, test_q0_worse, test_q1_worse,test_q2_worse, test_oracle_worse = [],[],[],[],[]
        test_correct, test_q0_correct, test_q1_correct, test_q2_correct, test_oracle_correct = [],[],[],[],[]
        # test the agent
        agent.epsilon = 0
        for test_serial, test_id in enumerate(test_ids):
            obs = user.initialize_state(test_id)
            ignore_questions = []
            q_done = False
            stop = False
            n_round = 0
            print('-------- test conversation %.0f/%.0f --------' % (test_serial + 1, len(test_ids)))
            while not q_done:
                print('-------- round %.0f --------' % (n_round))
                if obs in memory.keys():
                    question, answer = memory[obs][0], memory[obs][1]
                
                else:
                    # get retriever results
                    with open(question_retriever_args.qa_file, 'w') as retrieverinput:
                        riwriter = csv.writer(retrieverinput, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        riwriter.writerow([obs, ['']])

                    retrieve(question_retriever_args, question_retriever, question_index_buffer_sz)
                    retrieve(answer_retriever_args, answer_retriever, answer_index_buffer_sz)

                    with open(question_retriever_args.out_file) as question_f:
                        jsondata = json.load(question_f)
                        retrieved_question = [ctx['text'] for ctx in jsondata[0]['ctxs']]
                    with open(answer_retriever_args.out_file) as answer_f:
                        jsondata = json.load(answer_f)
                        retrieved_answer = [ctx['text'] for ctx in jsondata[0]['ctxs']]

                    retrieved_question = [rq for rq in retrieved_question if rq not in ignore_questions]

                    with open('/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_question', 'w') as qrerankerinput:
                        for rq in retrieved_question:
                            qrerankerinput.write(rq)
                            qrerankerinput.write('\n')
                    with open('/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_answer', 'w') as arerankerinput:
                        for ra in retrieved_answer:
                            arerankerinput.write(ra)
                            arerankerinput.write('\n')

                    # get reranker results
                    question = Interactive.main(model = 'transformer/polyencoder', \
                        model_file = 'zoo:pretrained_transformers/model_poly/msdialogquestion',  \
                        encode_candidate_vecs = 'true', eval_candidates ='fixed',  \
                        fixed_candidates_path = '/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_question', \
                        force_fp16_tokens = True,\
                        human_input=obs, fixed_candidate_vecs = 'replace')
                    answer = Interactive.main(model = 'transformer/polyencoder', \
                        model_file = 'zoo:pretrained_transformers/model_poly/msdialoganswer',  \
                        encode_candidate_vecs = 'true', eval_candidates ='fixed',  \
                        fixed_candidates_path = '/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_answer', \
                        human_input=obs,  fixed_candidate_vecs = 'replace')
                    
                    memory[obs] = [question, answer]

                action = agent.choose_action(obs, question, answer)
                obs_, question_reward, q_done, good_question = user.update_state(test_id, obs, 1, question, answer, use_top_k = use_top_k)
                _, answer_reward, _, _ = user.update_state(test_id, obs, 0, question, answer, use_top_k = use_top_k)
                action_reward = [answer_reward, question_reward][action]
                print('action', action, 'answer reward', answer_reward, 'question reward', question_reward, 'q done', q_done)

                if not q_done:
                    ignore_questions.append(good_question)
                    if obs_ in memory.keys():
                        question_, answer_ = memory[obs_][0], memory[obs_][1]
                    else:
                        # get retriever results
                        with open(question_retriever_args.qa_file, 'w') as retrieverinput:
                            riwriter = csv.writer(retrieverinput, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            riwriter.writerow([obs_, ['']])

                        retrieve(question_retriever_args, question_retriever, question_index_buffer_sz)
                        retrieve(answer_retriever_args, answer_retriever, answer_index_buffer_sz)

                        with open(answer_retriever_args.out_file) as question_f:
                            jsondata = json.load(question_f)
                            retrieved_question = [ctx['text'] for ctx in jsondata[0]['ctxs']]
                        with open(answer_retriever_args.out_file) as answer_f:
                            jsondata = json.load(answer_f)
                            retrieved_answer = [ctx['text'] for ctx in jsondata[0]['ctxs']]

                        retrieved_question = [rq for rq in retrieved_question if rq not in ignore_questions]

                        with open('/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_question', 'w') as qrerankerinput:
                            for rq in retrieved_question:
                                qrerankerinput.write(rq)
                                qrerankerinput.write('\n')
                        with open('/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_answer', 'w') as arerankerinput:
                            for ra in retrieved_answer:
                                arerankerinput.write(ra)
                                arerankerinput.write('\n')

                        # get reranker results
                        question_ = Interactive.main(model = 'transformer/polyencoder', \
                            model_file = 'zoo:pretrained_transformers/model_poly/msdialogquestion',  \
                            encode_candidate_vecs = 'true', eval_candidates ='fixed',  \
                            fixed_candidates_path = '/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_question', \
                            force_fp16_tokens = True,\
                            human_input=obs_, fixed_candidate_vecs = 'replace')
                        answer_ = Interactive.main(model = 'transformer/polyencoder', \
                            model_file = 'zoo:pretrained_transformers/model_poly/msdialoganswer',  \
                            encode_candidate_vecs = 'true', eval_candidates ='fixed',  \
                            fixed_candidates_path = '/raid/zhenduow/conversationalQA/data/' + exp_name + '/retrieved_answer', \
                            human_input=obs_,  fixed_candidate_vecs = 'replace')
                        
                        memory[obs_] = [question_, answer_]

                else:
                    question_, answer_ = None, None

                # evaluation
                if (action == 0 or (action == 1 and question_reward == cq_penalty)) and not stop:
                    stop = True
                    test_scores.append(answer_reward if action == 0 else 0)
                    if action == 0 and answer_reward == 1.0:
                        test_correct.append(test_id)
                    test_worse.append(1 if (action == 0 and answer_reward < float(1/use_top_k) and question_reward == cq_reward) \
                        or (action == 1 and answer_reward > 0 and question_reward == cq_penalty) else 0)
                if n_round == 0:
                    test_q0_scores.append(answer_reward)
                    test_q0_worse.append(1 if answer_reward < float(1/use_top_k) and question_reward == cq_reward else 0)
                    if answer_reward == 1:
                        test_q0_correct.append(test_id)
                    if q_done:
                        test_q1_scores.append(0)
                        test_q2_scores.append(0)
                        test_q1_worse.append(1)
                        test_q2_worse.append(1)
                elif n_round == 1:
                    test_q1_scores.append(answer_reward)
                    test_q1_worse.append(1 if answer_reward < float(1/use_top_k) and question_reward == cq_reward else 0)
                    if answer_reward == 1:
                        test_q1_correct.append(test_id)
                    if q_done:
                        test_q2_scores.append(0)
                        test_q2_worse.append(1)
                elif n_round == 2:
                    test_q2_scores.append(answer_reward)
                    test_q2_worse.append(1 if answer_reward < float(1/use_top_k) and question_reward == cq_reward else 0)
                    if answer_reward == 1:
                        test_q2_correct.append(test_id)

                n_round += 1
                obs = obs_

        assert len(test_scores) == len(test_q0_scores)
        assert len(test_q1_scores) == len(test_q0_scores)
        assert len(test_q1_scores) == len(test_q2_scores)
        for oi in range(len(test_scores)):
            test_oracle_scores.append(max(test_q0_scores[oi], test_q1_scores[oi], test_q2_scores[oi]))
            test_oracle_worse.append(min(test_q0_scores[oi], test_q1_scores[oi], test_q2_scores[oi]))
        test_oracle_correct = list(set(test_correct + test_q0_correct + test_q2_correct))

        print("Test epoch %.0f, acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (i, np.mean([1 if score == 1 else 0 for score in test_scores]), np.mean(test_scores), np.mean(test_worse)))
        print("q0 acc %.0f, avgmrr %.6f, worse decisions %.0f" % 
            (np.mean([1 if score == 1 else 0 for score in test_q0_scores]), np.mean(test_q0_scores), np.mean(test_q0_worse)))
        print("q1 acc %.0f, avgmrr %.6f, worse decisions %.0f" % 
            (np.mean([1 if score == 1 else 0 for score in test_q1_scores]), np.mean(test_q1_scores), np.mean(test_q1_worse)))
        print("q2 acc %.0f, avgmrr %.6f, worse decisions %.0f" % 
            (np.mean([1 if score == 1 else 0 for score in test_q2_scores]), np.mean(test_q2_scores), np.mean(test_q2_worse)))
        print("oracle acc %.0f, avgmrr %.6f, worse decisions %.0f" % 
            (np.mean([1 if score == 1 else 0 for score in test_oracle_scores]), np.mean(test_oracle_scores), np.mean(test_oracle_worse)))
        print(test_correct)
        print(test_q0_correct)
        print(test_q1_correct)
        print(test_q2_correct)
        print(test_oracle_correct)
