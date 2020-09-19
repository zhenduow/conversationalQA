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
cq_reward = 0.21
cq_penalty = cq_reward - 1
agent_gamma = -cq_penalty
train_iter = 20
use_top_k = 1
sampling_num = 9
exp_name = 'test'

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    train_dataset = ConversationDataset('data/MSDialog-Answer/train/')
    val_dataset = ConversationDataset('data/MSDialog-Answer/val/')
    test_dataset = ConversationDataset('data/MSDialog-Answer/test/')
    train_ids = list(train_dataset.conversations.keys())
    val_ids = list(val_dataset.conversations.keys())
    test_ids = list(test_dataset.conversations.keys())
    user = User({**train_dataset.conversations, **val_dataset.conversations, **test_dataset.conversations}, cq_reward = cq_reward, cq_penalty = cq_penalty)
    agent = Agent(lr=1e-4, input_dims = (2 + 4 * use_top_k) * observation_dim, n_actions=action_num, gamma = agent_gamma, weight_decay = 0.01)
    memory = {}
   
    for i in range(train_iter):
        train_scores, train_q0_scores, train_q1_scores, train_q2_scores, train_oracle_scores = [],[],[],[],[]
        train_worse, train_q0_worse, train_q1_worse, train_q2_worse, train_oracle_worse = [],[],[],[],[]
        train_correct, train_q0_correct, train_q1_correct, train_q2_correct, train_oracle_correct = [],[],[],[],[]
        for conv_serial, train_id in enumerate(train_ids):
            obs = user.initialize_state(train_id)
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

                    # sampling
                    retrieved_question = random.sample(train_dataset.responses_pool, sampling_num)
                    retrieved_answer = random.sample(train_dataset.answers_pool, sampling_num)
                    retrieved_question.append(train_dataset.conversations[train_id][2*n_round + 1])
                    retrieved_answer.append(train_dataset.conversations[train_id][-1])
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
                obs_, question_reward, q_done, good_question = user.update_state(train_id, obs, 1, question, answer, use_top_k = use_top_k)
                _, answer_reward, _, _ = user.update_state(train_id, obs, 0, question, answer, use_top_k = use_top_k)
                action_reward = [answer_reward, question_reward][action]
                print('action', action, 'answer reward', answer_reward, 'question reward', question_reward, 'q done', q_done)

                if not q_done:
                    ignore_questions.append(good_question)
                    if obs_ in memory.keys():
                        question_, answer_ = memory[obs_][0], memory[obs_][1]
                    else:
                        
                        # sampling    
                        retrieved_question = random.sample(train_dataset.responses_pool, sampling_num)
                        retrieved_answer = random.sample(train_dataset.answers_pool, sampling_num)
                        retrieved_question.append(train_dataset.conversations[train_id][2*n_round + 1])
                        retrieved_answer.append(train_dataset.conversations[train_id][-1])
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
                        train_correct.append(train_id) 
                    train_worse.append(1 if (action == 0 and answer_reward < float(1/use_top_k) and question_reward == cq_reward) \
                        or (action == 1 and answer_reward > 0 and question_reward == cq_penalty) else 0)
                if n_round == 0:
                    train_q0_scores.append(answer_reward)
                    train_q0_worse.append(1 if answer_reward < float(1/use_top_k) and question_reward == cq_reward else 0)
                    if answer_reward == 1:
                        train_q0_correct.append(train_id)
                    if q_done:
                        train_q1_scores.append(0)
                        train_q2_scores.append(0)
                        train_q1_worse.append(1)
                        train_q2_worse.append(1)
                elif n_round == 1:
                    train_q1_scores.append(answer_reward)
                    train_q1_worse.append(1 if answer_reward < float(1/use_top_k) and question_reward == cq_reward else 0)
                    if answer_reward == 1:
                        train_q1_correct.append(train_id)
                    if q_done:
                        train_q2_scores.append(0)
                        train_q2_worse.append(1)
                elif n_round == 2:
                    train_q2_scores.append(answer_reward)
                    train_q2_worse.append(1 if answer_reward < float(1/use_top_k) and question_reward == cq_reward else 0)
                    if answer_reward == 1:
                        train_q2_correct.append(train_id)

                obs = obs_
                n_round += 1


        assert len(train_scores) == len(train_q0_scores)
        assert len(train_q1_scores) == len(train_q0_scores)
        assert len(train_q1_scores) == len(train_q2_scores)
        for oi in range(len(train_scores)):
            train_oracle_scores.append(max(train_q0_scores[oi], train_q1_scores[oi], train_q2_scores[oi]))
            train_oracle_worse.append(min(train_q0_worse[oi], train_q1_worse[oi], train_q2_worse[oi]))
        train_oracle_correct = list(set(train_correct + train_q0_correct + train_q2_correct))

        print("Train epoch %.0f, acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (i, np.mean([1 if score == 1 else 0 for score in train_scores]), np.mean(train_scores), np.mean(train_worse)))
        print("q0 acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_q0_scores]), np.mean(train_q0_scores), np.mean(train_q0_worse)))
        print("q1 acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_q1_scores]), np.mean(train_q1_scores), np.mean(train_q1_worse)))
        print("q2 acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_q2_scores]), np.mean(train_q2_scores), np.mean(train_q2_worse)))
        print("oracle acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
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
                
                    # sampling
                    retrieved_question = random.sample(test_dataset.responses_pool, sampling_num)
                    retrieved_answer = random.sample(test_dataset.answers_pool, sampling_num)
                    retrieved_question.append(test_dataset.conversations[test_id][2*n_round + 1])
                    retrieved_answer.append(test_dataset.conversations[test_id][-1])
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
                        

                        # sampling
                        retrieved_question = random.sample(test_dataset.responses_pool, sampling_num)
                        retrieved_answer = random.sample(test_dataset.answers_pool, sampling_num)
                        retrieved_question.append(test_dataset.conversations[test_id][2*n_round + 1])
                        retrieved_answer.append(test_dataset.conversations[test_id][-1])
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
            test_oracle_worse.append(min(test_q0_worse[oi], test_q1_worse[oi], test_q2_worse[oi]))
        test_oracle_correct = list(set(test_correct + test_q0_correct + test_q2_correct))

        print("Test epoch %.0f, acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (i, np.mean([1 if score == 1 else 0 for score in test_scores]), np.mean(test_scores), np.mean(test_worse)))
        print("q0 acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_q0_scores]), np.mean(test_q0_scores), np.mean(test_q0_worse)))
        print("q1 acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_q1_scores]), np.mean(test_q1_scores), np.mean(test_q1_worse)))
        print("q2 acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_q2_scores]), np.mean(test_q2_scores), np.mean(test_q2_worse)))
        print("oracle acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_oracle_scores]), np.mean(test_oracle_scores), np.mean(test_oracle_worse)))
        print(test_correct)
        print(test_q0_correct)
        print(test_q1_correct)
        print(test_q2_correct)
        print(test_oracle_correct)
