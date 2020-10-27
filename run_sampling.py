from user import User
from dataset import ConversationDataset
from agent import Agent, BaseAgent, ScoreAgent, TextAgent
from bm25ranker import BM25Ranker
import logging
import numpy as np
import random
import json
import csv
import torch
import OpenMatch as om
from transformers import AutoTokenizer
from scipy.special import softmax
import sys
from parlai.scripts.interactive import Interactive, rerank
from copy import deepcopy
import argparse
observation_dim = 768
action_num = 2
cq_reward = 0.21
cq_penalty = cq_reward - 1
agent_gamma = -cq_penalty
train_iter = 100
use_top_k = 1
sampling_num = 99
ranker_name = 'Bi' # Poly, Bert, KNRM, Bi

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    train_dataset = ConversationDataset('data/MSDialog-Answer/train/')
    val_dataset = ConversationDataset('data/MSDialog-Answer/val/')
    test_dataset = ConversationDataset('data/MSDialog-Answer/test/')
    train_ids = list(train_dataset.conversations.keys())
    val_ids = list(val_dataset.conversations.keys())
    test_ids = list(test_dataset.conversations.keys())
    user = User({**train_dataset.conversations, **val_dataset.conversations, **test_dataset.conversations}, cq_reward = cq_reward, cq_penalty = cq_penalty)
    agent = Agent(lr=1e-4, input_dims = (3 + use_top_k) * observation_dim + 1 + use_top_k, top_k = use_top_k, n_actions=action_num, gamma = agent_gamma, weight_decay = 0.01)
    score_agent = ScoreAgent(lr = 1e-4, input_dims = 1 + use_top_k, top_k = use_top_k, n_actions=action_num, gamma = agent_gamma, weight_decay = 0.0)
    text_agent = TextAgent(lr = 1e-4, input_dims = (3 + use_top_k) * observation_dim, top_k = use_top_k, n_actions=action_num, gamma = agent_gamma, weight_decay = 0.01)
    base_agent = BaseAgent(lr=1e-4, input_dims = 2 * observation_dim, n_actions = 2, weight_decay = 0.01)
    memory = {}
    
    # create rerankers
    question_reranker = Interactive.main(model = 'transformer/polyencoder', \
                        model_file = 'zoo:pretrained_transformers/model_poly/msdialogquestion',  \
                        encode_candidate_vecs = False,  eval_candidates = 'inline', interactive_candidates = 'inline',
                        return_cand_scores = True)
    answer_reranker = Interactive.main(model = 'transformer/polyencoder', \
                        model_file = 'zoo:pretrained_transformers/model_poly/msdialoganswer',  \
                        encode_candidate_vecs = False,  eval_candidates = 'inline', interactive_candidates = 'inline',
                        return_cand_scores = True)
    
    bi_question_reranker = Interactive.main(model = 'transformer/biencoder', \
                        model_file = 'zoo:pretrained_transformers/model_bi/bimsdialogquestion',  \
                        encode_candidate_vecs = False,  eval_candidates = 'inline', interactive_candidates = 'inline',
                        return_cand_scores = True)
    bi_answer_reranker = Interactive.main(model = 'transformer/biencoder', \
                        model_file = 'zoo:pretrained_transformers/model_bi/bimsdialoganswer',  \
                        encode_candidate_vecs = False,  eval_candidates = 'inline', interactive_candidates = 'inline',
                        return_cand_scores = True)

    # initialize BertRanker
    '''
    BertTokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    BertQuestionRanker = om.models.Bert(pretrained = 'bert-base-uncased')
    BertAnswerRanker = om.models.Bert(pretrained = 'bert-base-uncased')
    state_dict = torch.load('./OpenMatch/checkpoints/bertquestion.bin')
    st = {}
    for k in state_dict:
        if k.startswith('bert'):
            st['_model'+k[len('bert'):]] = state_dict[k]
        elif k.startswith('classifier'):
            st['_dense'+k[len('classifier'):]] = state_dict[k]
        else:
            st[k] = state_dict[k]
    BertQuestionRanker.load_state_dict(st)
    state_dict = torch.load('./OpenMatch/checkpoints/bertanswer.bin')
    st = {}
    for k in state_dict:
        if k.startswith('bert'):
            st['_model'+k[len('bert'):]] = state_dict[k]
        elif k.startswith('classifier'):
            st['_dense'+k[len('classifier'):]] = state_dict[k]
        else:
            st[k] = state_dict[k]
    BertAnswerRanker.load_state_dict(st)
    # initialize KNRM 
    KNRMTokenizer = om.data.tokenizers.WordTokenizer(pretrained="./data/glove.6B.300d.txt")
    KNRMQuestionRanker = om.models.KNRM(vocab_size=KNRMTokenizer.get_vocab_size(),
                        embed_dim=KNRMTokenizer.get_embed_dim(),
                        embed_matrix=KNRMTokenizer.get_embed_matrix())
    state_dict = torch.load('./OpenMatch/checkpoints/knrmquestion.bin')
    KNRMQuestionRanker.load_state_dict(state_dict)
    KNRMAnswerRanker = om.models.KNRM(vocab_size=KNRMTokenizer.get_vocab_size(),
                        embed_dim=KNRMTokenizer.get_embed_dim(),
                        embed_matrix=KNRMTokenizer.get_embed_matrix())
    state_dict = torch.load('./OpenMatch/checkpoints/knrmanswer.bin')
    KNRMAnswerRanker.load_state_dict(state_dict)
    '''

    for i in range(train_iter):
        train_scores, train_q0_scores, train_q1_scores, train_q2_scores, train_oracle_scores, train_base_scores, train_score_scores, train_text_scores = [],[],[],[],[],[],[],[]
        train_worse, train_q0_worse, train_q1_worse, train_q2_worse, train_oracle_worse, train_base_worse, train_score_worse, train_text_worse = [],[],[],[],[],[],[],[]
        train_correct, train_q0_correct, train_q1_correct, train_q2_correct, train_oracle_correct, train_base_correct, train_score_correct,train_text_correct = [],[],[],[],[],[],[],[]
        for conv_serial, train_id in enumerate(train_ids):
            obs = user.initialize_state(train_id)
            ignore_questions = []
            n_round = 0
            q_done = False
            stop, base_stop, score_stop, text_stop = False,False,False,False
            print('-------- train conversation %.0f/%.0f --------' % (conv_serial + 1, len(train_ids)))
            while not q_done:
                print('-------- round %.0f --------' % (n_round))
                if obs in memory.keys():
                    question, answer, question_scores, answer_scores = memory[obs][0], memory[obs][1], memory[obs][2], memory[obs][3]
                else:

                    # sampling
                    retrieved_question = random.sample(train_dataset.responses_pool, sampling_num)
                    retrieved_answer = random.sample(train_dataset.answers_pool, sampling_num)
                    retrieved_question.append(train_dataset.conversations[train_id][2*n_round + 1])
                    retrieved_answer.append(train_dataset.conversations[train_id][-1])
                    retrieved_question = [rq for rq in retrieved_question if rq not in ignore_questions]

                    # get reranker results   
                    if ranker_name == 'Poly': 
                        obs_query, obs_context = '[SEP]'.join(obs.split('[SEP]')[:2]), ('[SEP]'.join(obs.split('[SEP]')[2:])).strip()
                        question, question_scores = rerank(question_reranker, human_input = obs_query, history = obs_context, \
                            sampled_candidates = retrieved_question)
                        answer, answer_scores = rerank(answer_reranker, human_input = obs_query, history = obs_context, \
                            sampled_candidates = retrieved_answer)

                    elif ranker_name == 'Bi': 
                        obs_query, obs_context = '[SEP]'.join(obs.split('[SEP]')[:2]), ('[SEP]'.join(obs.split('[SEP]')[2:])).strip()
                        question, question_scores = rerank(bi_question_reranker, human_input = obs_query, history = obs_context, \
                            sampled_candidates = retrieved_question)
                        answer, answer_scores = rerank(bi_answer_reranker, human_input = obs_query, history = obs_context, \
                            sampled_candidates = retrieved_answer)

                    '''
                    elif ranker_name == 'Bert':
                        question, answer, question_scores, answer_scores = [],[],[],[]
                        for cand in retrieved_question:
                            input_ids = BertTokenizer.encode(obs, cand)
                            input_ids = input_ids[:512]
                            ranking_score, ranking_features = BertQuestionRanker(torch.tensor(input_ids).unsqueeze(0))
                            question.append(cand)
                            question_scores.append(ranking_score[0].item())
                        question_scores = softmax(question_scores)
                        question = [q for _,q in sorted(zip(question_scores,question))]
                        question_scores = [s for s,_ in sorted(zip(question_scores,question))]
                        
                        for cand in retrieved_answer:
                            input_ids = BertTokenizer.encode(obs, cand)
                            input_ids = input_ids[:512]
                            ranking_score, ranking_features = BertAnswerRanker(torch.tensor(input_ids).unsqueeze(0))
                            answer.append(cand)
                            answer_scores.append(ranking_score[0].item())
                        answer_scores = softmax(answer_scores)
                        answer = [a for _,a in sorted(zip(answer_scores,answer))]
                        answer_scores = [s for s,_ in sorted(zip(answer_scores,answer))]
                        
                    elif ranker_name == 'KNRM':
                        question, answer, question_scores, answer_scores = [],[],[],[]
                        query_ids, query_masks = KNRMTokenizer.process(obs, max_len=16)
                        for cand in retrieved_question:
                            cand_ids, cand_masks = KNRMTokenizer.process(cand, max_len=128)
                            ranking_score, ranking_features = KNRMQuestionRanker(torch.tensor(query_ids).unsqueeze(0),
                                                                    torch.tensor(query_masks).unsqueeze(0),
                                                                    torch.tensor(cand_ids).unsqueeze(0),
                                                                    torch.tensor(cand_masks).unsqueeze(0))
                            question.append(cand)
                            question_scores.append(ranking_score[0].item())
                        question_scores = softmax(question_scores)
                        question = [q for _,q in sorted(zip(question_scores,question))]
                        question_scores = [s for s,_ in sorted(zip(question_scores,question))]
                        
                        for cand in retrieved_answer:
                            cand_ids, cand_masks = KNRMTokenizer.process(cand, max_len=128)
                            ranking_score, ranking_features = KNRMAnswerRanker(torch.tensor(query_ids).unsqueeze(0),
                                                                    torch.tensor(query_masks).unsqueeze(0),
                                                                    torch.tensor(cand_ids).unsqueeze(0),
                                                                    torch.tensor(cand_masks).unsqueeze(0))
                            answer.append(cand)
                            answer_scores.append(ranking_score[0].item())
                        answer_scores = softmax(answer_scores)
                        answer = [a for _,a in sorted(zip(answer_scores,answer))]
                        answer_scores = [s for s,_ in sorted(zip(answer_scores,answer))]
                    '''
                    memory[obs] = [question, answer, question_scores, answer_scores]

                action = agent.choose_action(obs, question, answer, question_scores, answer_scores)
                base_action = base_agent.choose_action(obs)
                score_action = score_agent.choose_action(question_scores, answer_scores)
                text_action = text_agent.choose_action(obs, question, answer)
                
                obs_, question_reward, q_done, good_question = user.update_state(train_id, obs, 1, question, answer, use_top_k = use_top_k)
                _, answer_reward, _, _ = user.update_state(train_id, obs, 0, question, answer, use_top_k = use_top_k)
                action_reward = [answer_reward, question_reward][action]
                print('action', action, 'base_action', base_action, 'score_action', score_action,'text_action', text_action, 'answer reward', answer_reward, 'question reward', question_reward, 'q done', q_done)

                if not q_done:
                    ignore_questions.append(good_question)
                    if obs_ in memory.keys():
                        question_, answer_, question_scores_, answer_scores_ = memory[obs_][0], memory[obs_][1], memory[obs_][2], memory[obs_][3]
                    else:
                        
                        # sampling    
                        retrieved_question = random.sample(train_dataset.responses_pool, sampling_num)
                        retrieved_answer = random.sample(train_dataset.answers_pool, sampling_num)
                        retrieved_question.append(train_dataset.conversations[train_id][2*n_round + 1])
                        retrieved_answer.append(train_dataset.conversations[train_id][-1])
                        retrieved_question = [rq for rq in retrieved_question if rq not in ignore_questions]

                        # get reranker results
                        if ranker_name == 'Poly': 
                            obs_query_, obs_context_ = '[SEP]'.join(obs_.split('[SEP]')[:2]), ('[SEP]'.join(obs_.split('[SEP]')[2:])).strip()
                            question_, question_scores_ = rerank(question_reranker, human_input = obs_query_, history = obs_context_, \
                                sampled_candidates = retrieved_question)
                            answer_, answer_scores_ = rerank(answer_reranker, human_input = obs_query_, history = obs_context_, \
                                sampled_candidates = retrieved_answer)

                        elif ranker_name == 'Bi': 
                            obs_query_, obs_context_ = '[SEP]'.join(obs_.split('[SEP]')[:2]), ('[SEP]'.join(obs_.split('[SEP]')[2:])).strip()
                            question_, question_scores_ = rerank(bi_question_reranker, human_input = obs_query_, history = obs_context_, \
                                sampled_candidates = retrieved_question)
                            answer_, answer_scores_ = rerank(bi_answer_reranker, human_input = obs_query_, history = obs_context_, \
                                sampled_candidates = retrieved_answer)
                        
                        memory[obs_] = [question_, answer_, question_scores_, answer_scores_]

                else:
                    question_, answer_, question_scores_, answer_scores_ = None, None, None, None

                agent.joint_learn((obs, question, answer, question_scores, answer_scores), answer_reward, question_reward, (obs_, question_, answer_, question_scores_, answer_scores_))
                base_agent.learn(obs, 0 if (n_round + 1) == len(user.dataset[train_id])/2 else 1)
                #score_agent.joint_learn((obs, question, answer, question_scores, answer_scores), answer_reward, question_reward, (obs_, question_, answer_, question_scores_, answer_scores_))
                #text_agent.joint_learn((obs, question, answer, question_scores, answer_scores), answer_reward, question_reward, (obs_, question_, answer_, question_scores_, answer_scores_))

                # evaluation
                if (action == 0 or (action == 1 and question_reward == cq_penalty)) and not stop:
                    stop = True 
                    train_scores.append(answer_reward if action == 0 else 0)
                    if action == 0 and answer_reward == 1.0:
                        train_correct.append(train_id) 
                    train_worse.append(1 if (action == 0 and answer_reward < float(1/use_top_k) and question_reward == cq_reward) \
                        or (action == 1 and answer_reward > 0 and question_reward == cq_penalty) else 0)

                if (base_action == 0 or (base_action == 1 and question_reward == cq_penalty)) and not base_stop:
                    base_stop = True
                    train_base_scores.append(answer_reward if base_action == 0 else 0)
                    if base_action == 0 and answer_reward == 1.0:
                        train_base_correct.append(train_id)
                    train_base_worse.append(1 if (base_action == 0 and answer_reward < float(1/use_top_k) and question_reward == cq_reward) \
                        or (base_action == 1 and answer_reward > 0 and question_reward == cq_penalty) else 0)
                
                if (score_action == 0 or (score_action == 1 and question_reward == cq_penalty)) and not score_stop:
                    score_stop = True
                    train_score_scores.append(answer_reward if score_action == 0 else 0)
                    if score_action == 0 and answer_reward == 1.0:
                        train_score_correct.append(train_id)
                    train_score_worse.append(1 if (score_action == 0 and answer_reward < float(1/use_top_k) and question_reward == cq_reward) \
                        or (score_action == 1 and answer_reward > 0 and question_reward == cq_penalty) else 0)
                
                if (text_action == 0 or (text_action == 1 and question_reward == cq_penalty)) and not text_stop:
                    text_stop = True
                    train_text_scores.append(answer_reward if text_action == 0 else 0)
                    if text_action == 0 and answer_reward == 1.0:
                        train_text_correct.append(train_id)
                    train_text_worse.append(1 if (text_action == 0 and answer_reward < float(1/use_top_k) and question_reward == cq_reward) \
                        or (text_action == 1 and answer_reward > 0 and question_reward == cq_penalty) else 0)

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


        #assert len(train_scores) == len(train_q0_scores)
        #assert len(train_q1_scores) == len(train_q0_scores)
        #assert len(train_q1_scores) == len(train_q2_scores)
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
        print("base cq identifier acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_base_scores]), np.mean(train_base_scores), np.mean(train_base_worse)))
        print("score acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_score_scores]), np.mean(train_score_scores), np.mean(train_score_worse)))
        print("text acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_text_scores]), np.mean(train_text_scores), np.mean(train_text_worse)))

        print(train_correct)
        print(train_q0_correct)
        print(train_q1_correct)
        print(train_q2_correct)
        print(train_oracle_correct)
        print(train_base_correct)
        print(train_score_correct)
        print(train_text_correct)
        print(train_scores)
        print(train_base_scores)
        print(train_score_scores)
        print(train_text_scores)
        print("avg loss", np.mean(agent.loss_history))

        ## test
        test_scores, test_q0_scores, test_q1_scores, test_q2_scores, test_oracle_scores, test_base_scores, test_score_scores, test_text_scores = [],[],[],[],[],[],[],[]
        test_worse, test_q0_worse, test_q1_worse,test_q2_worse, test_oracle_worse, test_base_worse, test_score_worse, test_text_worse = [],[],[],[],[],[],[],[]
        test_correct, test_q0_correct, test_q1_correct, test_q2_correct, test_oracle_correct, test_base_correct, test_score_correct, test_text_correct = [],[],[],[],[],[],[],[]
        # test the agent
        agent.epsilon = 0
        for test_serial, test_id in enumerate(test_ids):
            obs = user.initialize_state(test_id)
            ignore_questions = []
            n_round = 0
            q_done = False
            stop, base_stop, score_stop, text_stop = False,False,False,False
            print('-------- test conversation %.0f/%.0f --------' % (test_serial + 1, len(test_ids)))
            while not q_done:
                print('-------- round %.0f --------' % (n_round))
                if obs in memory.keys():
                    question, answer, question_scores, answer_scores = memory[obs][0], memory[obs][1], memory[obs][2], memory[obs][3]
                
                else:
                    # sampling
                    retrieved_question = random.sample(test_dataset.responses_pool, sampling_num)
                    retrieved_answer = random.sample(test_dataset.answers_pool, sampling_num)
                    retrieved_question.append(test_dataset.conversations[test_id][2*n_round + 1])
                    retrieved_answer.append(test_dataset.conversations[test_id][-1])
                    retrieved_question = [rq for rq in retrieved_question if rq not in ignore_questions]

                    # get reranker results
                    if ranker_name == 'Poly': 
                        obs_query, obs_context = '[SEP]'.join(obs.split('[SEP]')[:2]), ('[SEP]'.join(obs.split('[SEP]')[2:])).strip()
                        question, question_scores = rerank(question_reranker, human_input = obs_query, history = obs_context, \
                            sampled_candidates = retrieved_question)
                        answer, answer_scores = rerank(answer_reranker, human_input = obs_query, history = obs_context, \
                            sampled_candidates = retrieved_answer)
                    
                    elif ranker_name == 'Bi': 
                        obs_query, obs_context = '[SEP]'.join(obs.split('[SEP]')[:2]), ('[SEP]'.join(obs.split('[SEP]')[2:])).strip()
                        question, question_scores = rerank(bi_question_reranker, human_input = obs_query, history = obs_context, \
                            sampled_candidates = retrieved_question)
                        answer, answer_scores = rerank(bi_answer_reranker, human_input = obs_query, history = obs_context, \
                            sampled_candidates = retrieved_answer)
                
                    memory[obs] = [question, answer, question_scores, answer_scores]

                action = agent.choose_action(obs, question, answer, question_scores, answer_scores)
                base_action = base_agent.choose_action(obs = obs)
                score_action = score_agent.choose_action(question_scores, answer_scores)
                text_action = text_agent.choose_action(obs, question, answer)
                obs_, question_reward, q_done, good_question = user.update_state(test_id, obs, 1, question, answer, use_top_k = use_top_k)
                _, answer_reward, _, _ = user.update_state(test_id, obs, 0, question, answer, use_top_k = use_top_k)
                action_reward = [answer_reward, question_reward][action]
                print('action', action, 'base_action', base_action, 'score_action', score_action,'text_action', text_action,'answer reward', answer_reward, 'question reward', question_reward, 'q done', q_done)

                if not q_done:
                    ignore_questions.append(good_question)
                    if obs_ in memory.keys():
                        question_, answer_, question_scores_, answer_scores_ = memory[obs_][0], memory[obs_][1], memory[obs_][2], memory[obs_][3]
                    else:
                        # sampling
                        retrieved_question = random.sample(test_dataset.responses_pool, sampling_num)
                        retrieved_answer = random.sample(test_dataset.answers_pool, sampling_num)
                        retrieved_question.append(test_dataset.conversations[test_id][2*n_round + 1])
                        retrieved_answer.append(test_dataset.conversations[test_id][-1])
                        retrieved_question = [rq for rq in retrieved_question if rq not in ignore_questions]

                        # get reranker results
                        if ranker_name == 'Poly': 
                            obs_query_, obs_context_ = '[SEP]'.join(obs_.split('[SEP]')[:2]), ('[SEP]'.join(obs_.split('[SEP]')[2:])).strip()
                            question_, question_scores_ = rerank(question_reranker, human_input = obs_query_, history = obs_context_, \
                                sampled_candidates = retrieved_question)
                            answer_, answer_scores_ = rerank(answer_reranker, human_input = obs_query_, history = obs_context_, \
                                sampled_candidates = retrieved_answer)
                        elif ranker_name == 'Bi': 
                            obs_query_, obs_context_ = '[SEP]'.join(obs_.split('[SEP]')[:2]), ('[SEP]'.join(obs_.split('[SEP]')[2:])).strip()
                            question_, question_scores_ = rerank(bi_question_reranker, human_input = obs_query_, history = obs_context_, \
                                sampled_candidates = retrieved_question)
                            answer_, answer_scores_ = rerank(bi_answer_reranker, human_input = obs_query_, history = obs_context_, \
                                sampled_candidates = retrieved_answer)

                        memory[obs_] = [question_, answer_, question_scores_, answer_scores_]

                else:
                    question_, answer_, question_scores_, answer_scores_ = None, None, None, None

                # evaluation
                if (action == 0 or (action == 1 and question_reward == cq_penalty)) and not stop:
                    stop = True
                    test_scores.append(answer_reward if action == 0 else 0)
                    if action == 0 and answer_reward == 1.0:
                        test_correct.append(test_id)
                    test_worse.append(1 if (action == 0 and answer_reward < float(1/use_top_k) and question_reward == cq_reward) \
                        or (action == 1 and answer_reward > 0 and question_reward == cq_penalty) else 0)

                if (base_action == 0 or (base_action == 1 and question_reward == cq_penalty)) and not base_stop:
                    base_stop = True
                    test_base_scores.append(answer_reward if base_action == 0 else 0)
                    if base_action == 0 and answer_reward == 1.0:
                        test_base_correct.append(test_id)
                    test_base_worse.append(1 if (base_action == 0 and answer_reward < float(1/use_top_k) and question_reward == cq_reward) \
                        or (base_action == 1 and answer_reward > 0 and question_reward == cq_penalty) else 0)

                if (score_action == 0 or (score_action == 1 and question_reward == cq_penalty)) and not score_stop:
                    score_stop = True
                    test_score_scores.append(answer_reward if score_action == 0 else 0)
                    if score_action == 0 and answer_reward == 1.0:
                        test_score_correct.append(test_id)
                    test_score_worse.append(1 if (score_action == 0 and answer_reward < float(1/use_top_k) and question_reward == cq_reward) \
                        or (score_action == 1 and answer_reward > 0 and question_reward == cq_penalty) else 0)
                
                if (text_action == 0 or (text_action == 1 and question_reward == cq_penalty)) and not text_stop:
                    text_stop = True
                    test_text_scores.append(answer_reward if text_action == 0 else 0)
                    if text_action == 0 and answer_reward == 1.0:
                        test_text_correct.append(test_id)
                    test_text_worse.append(1 if (text_action == 0 and answer_reward < float(1/use_top_k) and question_reward == cq_reward) \
                        or (text_action == 1 and answer_reward > 0 and question_reward == cq_penalty) else 0)

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

        #assert len(test_scores) == len(test_q0_scores)
        #assert len(test_q1_scores) == len(test_q0_scores)
        #assert len(test_q1_scores) == len(test_q2_scores)
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
        print("base cq identifier acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_base_scores]), np.mean(test_base_scores), np.mean(test_base_worse)))
        print("score acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_score_scores]), np.mean(test_score_scores), np.mean(test_score_worse)))
        print("text acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_text_scores]), np.mean(test_text_scores), np.mean(test_text_worse)))
        print(test_correct)
        print(test_q0_correct)
        print(test_q1_correct)
        print(test_q2_correct)
        print(test_oracle_correct)
        print(test_base_correct)
        print(test_score_correct)
        print(test_text_correct)
        print(test_scores)
        print(test_base_scores)
        print(test_score_scores)
        print(test_text_scores)
