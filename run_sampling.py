from user import User
from dataset import ConversationDataset
from agent import Agent, BaseAgent, ScoreAgent, TextAgent
import logging
import numpy as np
import random
import json
import resource
import csv
import os
import torch as T
from transformers import AutoTokenizer, AutoModel
from scipy.special import softmax
import sys
import time
from parlai.scripts.interactive import Interactive, rerank
from copy import deepcopy
import argparse
import psutil
observation_dim = 768
action_num = 2
cq_reward = 0.11
cq_penalty = cq_reward - 1
agent_gamma = -cq_penalty
train_iter = 50
batch_size = 100
max_round = 5 # max conversation round
max_train_size = 10000
max_test_size = int(0.25*max_train_size)

def limit_memory(maxsize): 
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard)) 

def generate_embedding_no_grad(text, tokenizer, embedding_model):
    with T.no_grad():
        tokenized_context_ = T.tensor([tokenizer.encode(text, add_special_tokens=True)])
        context_embedding_ = T.squeeze(embedding_model(tokenized_context_)[0])[0] 
        del tokenized_context_
        return context_embedding_

def read_from_memory(query, context, memory):
    return memory[query]['embedding'], memory[query][context]['embedding'],\
        memory[query][context]['questions'], memory[query][context]['answers'],\
        memory[query][context]['questions_embeddings'],memory[query][context]['answers_embeddings'],\
        memory[query][context]['questions_scores'], memory[query][context]['answers_scores']

def save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model):
    if query not in memory.keys():
        memory[query] = {}
        with T.no_grad():
            tokenized_query = T.tensor([tokenizer.encode(query, add_special_tokens=True)])
            memory[query]['embedding'] = T.squeeze(embedding_model(tokenized_query)[0])[0]
    
    memory[query][context] = {}
    with T.no_grad():
        memory[query][context]['embedding'] = T.squeeze(embedding_model(T.tensor([tokenizer.encode(context, add_special_tokens=True)]))[0])[0]
        memory[query][context]['questions_embeddings'] = [T.squeeze(embedding_model(T.tensor([tokenizer.encode(questions[i], add_special_tokens=True)]))[0])[0] for i in range(3)]
        memory[query][context]['answers_embeddings'] = [T.squeeze(embedding_model(T.tensor([tokenizer.encode(answers[0], add_special_tokens=True)]))[0])[0]]
    memory[query][context]['questions'] = questions
    memory[query][context]['answers'] = answers
    memory[query][context]['questions_scores'] = T.tensor(questions_scores)
    memory[query][context]['answers_scores'] = T.tensor(answers_scores)
    return memory

def generate_batch_question_candidates(batch, conversation_id, ignore_questions, total_candidates):
    positives = [batch['conversations'][conversation_id][turn_id] for turn_id in range(len(batch['conversations'][conversation_id])) if turn_id % 2 == 1 and turn_id != len(batch['conversations'][conversation_id])-1]
    filtered_positives = [cand for cand in positives if cand not in ignore_questions]
    negatives = [response for response in batch['responses_pool'] if response not in positives][:total_candidates - len(filtered_positives)]
    return filtered_positives + negatives

def generate_batch_answer_candidates(batch, conversation_id, total_candidates):
    positives = [batch['conversations'][conversation_id][-1]]
    negatives = [answer for answer in batch['answers_pool'] if answer not in positives][:total_candidates - len(positives)] 
    return positives + negatives


def main(args):
    logging.getLogger().setLevel(logging.INFO)
    limit_memory(1e11)

    random.seed(2020)
    if args.cv != -1:
        train_dataset = ConversationDataset('data/' + args.dataset_name + '-Complete/train' + str(args.cv) + '/', batch_size, max_train_size)
        test_dataset = ConversationDataset('data/' + args.dataset_name + '-Complete/test' + str(args.cv) + '/', batch_size, max_test_size)
    else:
        train_dataset = ConversationDataset('data/' + args.dataset_name + '-Complete/train/', batch_size, max_train_size)
        test_dataset = ConversationDataset('data/' + args.dataset_name + '-Complete/test/' , batch_size, max_test_size)
    train_size = sum([len(b['conversations'].keys()) for b in train_dataset.batches]) 
    test_size = sum([len(b['conversations'].keys()) for b in test_dataset.batches]) 
    agent = Agent(lr = 1e-4, input_dims = (3 + args.topn) * observation_dim + 1 + args.topn, top_k = args.topn, n_actions=action_num, gamma = agent_gamma, weight_decay = 0.01)
    score_agent = ScoreAgent(lr = 1e-4, input_dims = 1 + args.topn, top_k = args.topn, n_actions=action_num, gamma = agent_gamma, weight_decay = 0.0)
    text_agent = TextAgent(lr = 1e-4, input_dims = (3 + args.topn) * observation_dim, top_k = args.topn, n_actions=action_num, gamma = agent_gamma, weight_decay = 0.01)
    base_agent = BaseAgent(lr = 1e-4, input_dims = 2 * observation_dim, n_actions = 2, weight_decay = 0.01)
    
    if args.dataset_name == 'MSDialog':
        reranker_prefix = ''
    elif args.dataset_name == 'UDC':
        reranker_prefix = 'udc'
    elif args.dataset_name == 'Opendialkg':
        reranker_prefix = 'open'
    # create rerankers
    if args.reranker_name == 'Poly':
        question_reranker = Interactive.main(model = 'transformer/polyencoder', \
                            model_file = 'zoo:pretrained_transformers/model_poly/' + reranker_prefix + 'question',  \
                            encode_candidate_vecs = False,  eval_candidates = 'inline', interactive_candidates = 'inline',
                            return_cand_scores = True)
        answer_reranker = Interactive.main(model = 'transformer/polyencoder', \
                            model_file = 'zoo:pretrained_transformers/model_poly/' + reranker_prefix + 'answer',  \
                            encode_candidate_vecs = False,  eval_candidates = 'inline', interactive_candidates = 'inline',
                            return_cand_scores = True)
        print("Loading rerankers:", 'model_poly/' + reranker_prefix + 'answer', 'model_poly/' + reranker_prefix + 'question')
    elif args.reranker_name == 'Bi':
        bi_question_reranker = Interactive.main(model = 'transformer/biencoder', \
                            model_file = 'zoo:pretrained_transformers/model_bi/' + reranker_prefix + 'question',  \
                            encode_candidate_vecs = False,  eval_candidates = 'inline', interactive_candidates = 'inline',
                            return_cand_scores = True)
        bi_answer_reranker = Interactive.main(model = 'transformer/biencoder', \
                            model_file = 'zoo:pretrained_transformers/model_bi/' + reranker_prefix + 'answer',  \
                            encode_candidate_vecs = False,  eval_candidates = 'inline', interactive_candidates = 'inline',
                            return_cand_scores = True)
        print("Loading rerankers:", 'model_bi/' + reranker_prefix + 'answer', 'model_bi/' + reranker_prefix + 'question')

    # embedding model
    tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
    embedding_model = AutoModel.from_pretrained('xlnet-base-cased')
    '''
    local_vars = list(locals().items())
    for var, obj in local_vars:
        print(var, get_size(obj))
    '''

    if not os.path.exists(args.dataset_name + '_experiments/embedding_cache/'):
        os.makedirs(args.dataset_name + '_experiments/embedding_cache/')
    if not os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name ):
        os.makedirs(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name )
    if args.cv != -1:
        if not os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + str(args.cv)):
            os.makedirs(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + str(args.cv))
            os.makedirs(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + str(args.cv) + '/train')
            os.makedirs(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + str(args.cv) + '/test')
    else:
        if not os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/train' ):
            os.makedirs(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/train')
        if not os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/test' ):
            os.makedirs(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/test' )

    for i in range(train_iter):
        train_scores, train_q0_scores, train_q1_scores, train_q2_scores, train_oracle_scores, train_base_scores, train_score_scores, train_text_scores = [],[],[],[],[],[],[],[]
        train_worse, train_q0_worse, train_q1_worse, train_q2_worse, train_oracle_worse, train_base_worse, train_score_worse, train_text_worse = [],[],[],[],[],[],[],[]
        #train_correct, train_q0_correct, train_q1_correct, train_q2_correct, train_oracle_correct, train_base_correct, train_score_correct,train_text_correct = [],[],[],[],[],[],[],[]
        for batch_serial, batch in enumerate(train_dataset.batches):
            print(dict(psutil.virtual_memory()._asdict()))
            if args.cv != -1:
                if os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + str(args.cv) + '/train/memory.batchsave' + str(batch_serial)):
                    with T.no_grad():
                        memory = T.load(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + str(args.cv) + '/train/memory.batchsave' + str(batch_serial))
                else:
                    memory = {}
            else:
                if os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/train/memory.batchsave' + str(batch_serial)):
                    with T.no_grad():
                        memory = T.load(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/train/memory.batchsave' + str(batch_serial))
                else:
                    memory = {}
            train_ids = list(batch['conversations'].keys())
            user = User(batch['conversations'], cq_reward = cq_reward, cq_penalty = cq_penalty)
            for conv_serial, train_id in enumerate(train_ids):
                query = user.initialize_state(train_id)
                if query == '': # UDC dataset has some weird stuff
                    continue
                context = ''
                ignore_questions = []
                n_round = 0
                patience_used = 0
                q_done = False
                stop, base_stop, score_stop, text_stop = False,False,False,False
                print('-------- train batch %.0f conversation %.0f/%.0f --------' % (batch_serial, batch_size*(batch_serial) + conv_serial + 1, train_size))
                
                while not q_done:
                    total_tic = time.perf_counter()
                    print('-------- round %.0f --------' % (n_round))
                    if query in memory.keys():
                        if context not in memory[query].keys():
                            # sampling
                            question_candidates = generate_batch_question_candidates(batch, train_id, ignore_questions, batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, train_id, batch_size)
                            # get reranker results   
                            if args.reranker_name == 'Poly': 
                                questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                                answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                            elif args.reranker_name == 'Bi': 
                                questions, questions_scores = rerank(bi_question_reranker, query, context, question_candidates)
                                answers, answers_scores = rerank(bi_answer_reranker, query, context, answer_candidates)

                            memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model)
                            
                    else:
                        # sampling
                        question_candidates = generate_batch_question_candidates(batch, train_id, ignore_questions, batch_size)
                        answer_candidates = generate_batch_answer_candidates(batch, train_id, batch_size)
                        # get reranker results   
                        if args.reranker_name == 'Poly': 
                            questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                            answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                        elif args.reranker_name == 'Bi': 
                            questions, questions_scores = rerank(bi_question_reranker, query, context, question_candidates)
                            answers, answers_scores = rerank(bi_answer_reranker, query, context, answer_candidates)
                        
                        memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model)
                    
                    query_embedding, context_embedding, questions, answers, questions_embeddings, answers_embeddings, questions_scores, answers_scores = read_from_memory(query, context, memory)
                    action = agent.choose_action(query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores)
                    base_action = base_agent.choose_action(query_embedding, context_embedding)
                    score_action = score_agent.choose_action(questions_scores, answers_scores)
                    text_action = text_agent.choose_action(query_embedding, context_embedding, questions_embeddings, answers_embeddings)

                    evaluation_tic = time.perf_counter()
                    #context_, question_reward, q_done, good_question, patience_this_turn = user.update_state(train_id, context, 1, questions, answers, use_top_k = args.topn - patience_used)
                    context_, question_reward, q_done, good_question, patience_this_turn = user.update_state(train_id, context, 1, questions, answers, use_top_k = args.topn)
                    patience_used = max(patience_used + patience_this_turn, args.topn)
                    _, answer_reward, _, _, _ = user.update_state(train_id, context, 0, questions, answers, use_top_k = args.topn - patience_used)
                    action_reward = [answer_reward, question_reward][action]
                    evaluation_toc = time.perf_counter()
                    print('action', action, 'base_action', base_action, 'score_action', score_action,'text_action', text_action, 'answer reward', answer_reward, 'question reward', question_reward, 'q done', q_done)

                    if n_round >= max_round:
                        q_done = True

                    if not q_done:
                        ignore_questions.append(good_question)
                        if context_ not in memory[query].keys():
                            # sampling    
                            question_candidates = generate_batch_question_candidates(batch, train_id, ignore_questions, batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, train_id, batch_size)

                            # get reranker results
                            if args.reranker_name == 'Poly': 
                                questions_, questions_scores_ = rerank(question_reranker, query, context_, question_candidates)
                                answers_, answers_scores_ = rerank(answer_reranker, query, context_, answer_candidates)
                            elif args.reranker_name == 'Bi': 
                                questions_, questions_scores_ = rerank(bi_question_reranker, query, context_, question_candidates)
                                answers_, answers_scores_ = rerank(bi_answer_reranker, query, context_, answer_candidates)
                            
                            memory = save_to_memory(query, context_, memory, questions_, answers_, questions_scores_, answers_scores_, tokenizer, embedding_model)
                        query_embedding, context_embedding_, questions_, answers_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_ = read_from_memory(query, context_, memory)

                    else:
                        context_embedding_ = generate_embedding_no_grad(context_, tokenizer, embedding_model)
                        questions_, answers_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_ = None, None, None, None, None, None

                    agent.joint_learn((query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores),\
                        answer_reward, question_reward,\
                        (query_embedding, context_embedding_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_))
                    base_agent.learn(query_embedding, context_embedding, 0 if (n_round + 1) == len(user.dataset[train_id])/2 else 1)
                    score_agent.joint_learn((questions_scores, answers_scores),\
                        answer_reward, question_reward,\
                        (questions_scores_, answers_scores_))
                    text_agent.joint_learn((query_embedding,context_embedding, questions_embeddings, answers_embeddings),\
                        answer_reward, question_reward,\
                        (query_embedding, context_embedding_, questions_embeddings_, answers_embeddings_))

                    # non-deterministic methods evaluation
                    if not stop:
                        train_worse.append(1 if (action == 0 and answer_reward < float(1/args.topn) and question_reward == cq_reward) \
                            or (action == 1  and question_reward == cq_penalty) else 0)
                        if (action == 0 or (action == 1 and question_reward == cq_penalty)):
                            stop = True 
                            train_scores.append(answer_reward if action == 0 else 0)
                            if action == 0 and answer_reward == 1.0:
                                #train_correct.append(train_id) 
                                pass
                        
                    if not base_stop:
                        train_base_worse.append(1 if (base_action == 0 and answer_reward < float(1/args.topn) and question_reward == cq_reward) \
                            or (base_action == 1  and question_reward == cq_penalty) else 0)
                        if (base_action == 0 or (base_action == 1 and question_reward == cq_penalty)):
                            base_stop = True
                            train_base_scores.append(answer_reward if base_action == 0 else 0)
                            if base_action == 0 and answer_reward == 1.0:
                                #train_base_correct.append(train_id)
                                pass
                        
                    if not score_stop:
                        train_score_worse.append(1 if (score_action == 0 and answer_reward < float(1/args.topn) and question_reward == cq_reward) \
                            or (score_action == 1  and question_reward == cq_penalty) else 0)
                        if (score_action == 0 or (score_action == 1 and question_reward == cq_penalty)):
                            score_stop = True
                            train_score_scores.append(answer_reward if score_action == 0 else 0)
                            if score_action == 0 and answer_reward == 1.0:
                                pass
                                #train_score_correct.append(train_id)
                        
                    if not text_stop:
                        train_text_worse.append(1 if (text_action == 0 and answer_reward < float(1/args.topn) and question_reward == cq_reward) \
                            or (text_action == 1  and question_reward == cq_penalty) else 0)
                        if (text_action == 0 or (text_action == 1 and question_reward == cq_penalty)):
                            text_stop = True
                            train_text_scores.append(answer_reward if text_action == 0 else 0)
                            if text_action == 0 and answer_reward == 1.0:
                                pass
                                #train_text_correct.append(train_id)
                        
                    # deterministic method evaluation
                    if n_round == 0:
                        train_q0_scores.append(answer_reward)
                        train_q0_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == cq_reward else 0)
                        if answer_reward == 1:
                            pass
                            #train_q0_correct.append(train_id)
                        if q_done:
                            train_q1_scores.append(0)
                            train_q2_scores.append(0)
                            train_q1_worse.append(1)
                            train_q2_worse.append(1)
                    elif n_round == 1:
                        train_q1_scores.append(answer_reward)
                        train_q1_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == cq_reward else 0)
                        if answer_reward == 1:
                            pass
                            #train_q1_correct.append(train_id)
                        if q_done:
                            train_q2_scores.append(0)
                            train_q2_worse.append(1)
                    elif n_round == 2:
                        train_q2_scores.append(answer_reward)
                        train_q2_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == cq_reward else 0)
                        if answer_reward == 1:
                            pass
                            #train_q2_correct.append(train_id)

                    context = context_
                    n_round += 1
                    total_toc = time.perf_counter()
            
            # save memory per batch
            if args.cv != -1:
                T.save(memory, args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + str(args.cv) + '/train/memory.batchsave' + str(batch_serial))
            else:
                T.save(memory, args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/train/memory.batchsave' + str(batch_serial))
            
            del memory
            T.cuda.empty_cache()

        for oi in range(len(train_scores)):
            train_oracle_scores.append(max(train_q0_scores[oi], train_q1_scores[oi], train_q2_scores[oi]))
            train_oracle_worse.append(min(train_q0_worse[oi], train_q1_worse[oi], train_q2_worse[oi]))
        #train_oracle_correct = list(set(train_correct + train_q0_correct + train_q2_correct))

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
        print("base acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_base_scores]), np.mean(train_base_scores), np.mean(train_base_worse)))
        print("score acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_score_scores]), np.mean(train_score_scores), np.mean(train_score_worse)))
        print("text acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in train_text_scores]), np.mean(train_text_scores), np.mean(train_text_worse)))

        print("avg loss", np.mean(agent.loss_history))

        ## test
        test_scores, test_q0_scores, test_q1_scores, test_q2_scores, test_oracle_scores, test_base_scores, test_score_scores, test_text_scores = [],[],[],[],[],[],[],[]
        test_worse, test_q0_worse, test_q1_worse,test_q2_worse, test_oracle_worse, test_base_worse, test_score_worse, test_text_worse = [],[],[],[],[],[],[],[]
        #test_correct, test_q0_correct, test_q1_correct, test_q2_correct, test_oracle_correct, test_base_correct, test_score_correct, test_text_correct = [],[],[],[],[],[],[],[]
        # test the agent
        agent.epsilon = 0
        
        for batch_serial, batch in enumerate(test_dataset.batches):
            if args.cv != -1:
                if os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + str(args.cv) + '/test/memory.batchsave' + str(batch_serial)):
                    with T.no_grad():
                        memory = T.load(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + str(args.cv) + '/test/memory.batchsave' + str(batch_serial))
                else:
                    memory = {}
            else:
                if os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/test/memory.batchsave' + str(batch_serial)):
                    with T.no_grad():
                        memory = T.load(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/test/memory.batchsave' + str(batch_serial))
                else:
                    memory = {}
            test_ids = list(batch['conversations'].keys())
            user = User(batch['conversations'], cq_reward = cq_reward, cq_penalty = cq_penalty)
            for conv_serial, test_id in enumerate(test_ids):
                query = user.initialize_state(test_id)
                if query == '': # UDC dataset has some weird stuff
                    continue
                context = ''
                ignore_questions = []
                n_round = 0
                patience_used = 0
                q_done = False
                stop, base_stop, score_stop, text_stop = False,False,False,False
                print('-------- test batch %.0f conversation %.0f/%.0f --------' % (batch_serial, batch_size*(batch_serial) + conv_serial + 1, test_size))
                while not q_done:
                    print('-------- round %.0f --------' % (n_round))
                    if query in memory.keys():
                        if context not in memory[query].keys():
                            # sampling
                            question_candidates = generate_batch_question_candidates(batch, test_id, ignore_questions, batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, test_id, batch_size)
                            # get reranker results   
                            if args.reranker_name == 'Poly': 
                                questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                                answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                            elif args.reranker_name == 'Bi': 
                                questions, questions_scores = rerank(bi_question_reranker, query, context, question_candidates)
                                answers, answers_scores = rerank(bi_answer_reranker, query, context, answer_candidates)

                            memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model)
                            
                    else:
                        # sampling
                        question_candidates = generate_batch_question_candidates(batch, test_id, ignore_questions, batch_size)
                        answer_candidates = generate_batch_answer_candidates(batch, test_id, batch_size)

                        # get reranker results
                        if args.reranker_name == 'Poly': 
                            questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                            answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                        elif args.reranker_name == 'Bi':
                            questions, questions_scores = rerank(bi_question_reranker, query, context, question_candidates)
                            answers, answers_scores = rerank(bi_answer_reranker, query, context, answer_candidates)
                    
                        memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model)
                    
                    query_embedding, context_embedding, questions, answers, questions_embeddings, answers_embeddings, questions_scores, answers_scores = read_from_memory(query, context, memory)
                    action = agent.choose_action(query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores)
                    base_action = base_agent.choose_action(query_embedding, context_embedding)
                    score_action = score_agent.choose_action(questions_scores, answers_scores)
                    text_action = text_agent.choose_action(query_embedding, context_embedding, questions_embeddings, answers_embeddings)
                    
                    #context_, question_reward, q_done, good_question, patience_this_turn = user.update_state(test_id, context, 1, questions, answers, use_top_k = args.topn - patience_used)
                    context_, question_reward, q_done, good_question, patience_this_turn = user.update_state(test_id, context, 1, questions, answers, use_top_k = args.topn)
                    patience_used = max(patience_used + patience_this_turn, args.topn)
                    _, answer_reward, _, _, _ = user.update_state(test_id, context, 0, questions, answers, use_top_k = args.topn - patience_used)
                    action_reward = [answer_reward, question_reward][action]
                    print('action', action, 'base_action', base_action, 'score_action', score_action,'text_action', text_action, 'answer reward', answer_reward, 'question reward', question_reward, 'q done', q_done)

                    if n_round >= max_round:
                        q_done = True

                    if not q_done:
                        ignore_questions.append(good_question)
                        if context_ not in memory[query].keys():
                            # sampling    
                            question_candidates = generate_batch_question_candidates(batch, test_id, ignore_questions, batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, test_id, batch_size)
                            # get reranker results
                            if args.reranker_name == 'Poly': 
                                questions_, questions_scores_ = rerank(question_reranker, query, context_, question_candidates)
                                answers_, answers_scores_ = rerank(answer_reranker, query, context_, answer_candidates)
                            elif args.reranker_name == 'Bi': 
                                questions_, questions_scores_ = rerank(bi_question_reranker, query, context_, question_candidates)
                                answers_, answers_scores_ = rerank(bi_answer_reranker, query, context_, answer_candidates)
                            
                            memory = save_to_memory(query, context_, memory, questions_, answers_, questions_scores_, answers_scores_, tokenizer, embedding_model)
                        query_embedding, context_embedding_, questions_, answers_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_ = read_from_memory(query, context_, memory)

                    # non-deterministic methods evaluation
                    if not stop:
                        test_worse.append(1 if (action == 0 and answer_reward < float(1/args.topn) and question_reward == cq_reward) \
                            or (action == 1  and question_reward == cq_penalty) else 0)
                        if (action == 0 or (action == 1 and question_reward == cq_penalty)):
                            stop = True
                            test_scores.append(answer_reward if action == 0 else 0)
                            if action == 0 and answer_reward == 1.0:
                                pass
                                #test_correct.append(test_id)
                        
                    if not base_stop:
                        test_base_worse.append(1 if (base_action == 0 and answer_reward < float(1/args.topn) and question_reward == cq_reward) \
                            or (base_action == 1  and question_reward == cq_penalty) else 0)
                        if (base_action == 0 or (base_action == 1 and question_reward == cq_penalty)):
                            base_stop = True
                            test_base_scores.append(answer_reward if base_action == 0 else 0)
                            if base_action == 0 and answer_reward == 1.0:
                                pass
                                #test_base_correct.append(test_id)
                    
                    if not score_stop:
                        test_score_worse.append(1 if (score_action == 0 and answer_reward < float(1/args.topn) and question_reward == cq_reward) \
                            or (score_action == 1  and question_reward == cq_penalty) else 0)
                        if (score_action == 0 or (score_action == 1 and question_reward == cq_penalty)):
                            score_stop = True
                            test_score_scores.append(answer_reward if score_action == 0 else 0)
                            if score_action == 0 and answer_reward == 1.0:
                                pass
                                #test_score_correct.append(test_id)
                        
                    if not text_stop:
                        test_text_worse.append(1 if (text_action == 0 and answer_reward < float(1/args.topn) and question_reward == cq_reward) \
                            or (text_action == 1  and question_reward == cq_penalty) else 0)
                        if (text_action == 0 or (text_action == 1 and question_reward == cq_penalty)):
                            text_stop = True
                            test_text_scores.append(answer_reward if text_action == 0 else 0)
                            if text_action == 0 and answer_reward == 1.0:
                                pass
                                #test_text_correct.append(test_id)
                        
                    
                    # deterministic methods evaluation
                    if n_round == 0:
                        test_q0_scores.append(answer_reward)
                        test_q0_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == cq_reward else 0)
                        if answer_reward == 1:
                            pass
                            #test_q0_correct.append(test_id)
                        if q_done:
                            test_q1_scores.append(0)
                            test_q2_scores.append(0)
                            test_q1_worse.append(1)
                            test_q2_worse.append(1)
                    elif n_round == 1:
                        test_q1_scores.append(answer_reward)
                        test_q1_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == cq_reward else 0)
                        if answer_reward == 1:
                            pass
                            #test_q1_correct.append(test_id)
                        if q_done:
                            test_q2_scores.append(0)
                            test_q2_worse.append(1)
                    elif n_round == 2:
                        test_q2_scores.append(answer_reward)
                        test_q2_worse.append(1 if answer_reward < float(1/args.topn) and question_reward == cq_reward else 0)
                        if answer_reward == 1:
                            pass
                            #test_q2_correct.append(test_id)

                    n_round += 1
                    context = context_
            
            # save batch cache
            if args.cv != -1:
                T.save(memory, args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + str(args.cv) + '/test/memory.batchsave' + str(batch_serial))
            else:
                T.save(memory, args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/test/memory.batchsave' + str(batch_serial))

            del memory
            T.cuda.empty_cache()

          
        for oi in range(len(test_scores)):
            test_oracle_scores.append(max(test_q0_scores[oi], test_q1_scores[oi], test_q2_scores[oi]))
            test_oracle_worse.append(min(test_q0_worse[oi], test_q1_worse[oi], test_q2_worse[oi]))
        #test_oracle_correct = list(set(test_correct + test_q0_correct + test_q2_correct))

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
        print("base acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_base_scores]), np.mean(test_base_scores), np.mean(test_base_worse)))
        print("score acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_score_scores]), np.mean(test_score_scores), np.mean(test_score_worse)))
        print("text acc %.6f, avgmrr %.6f, worse decisions %.6f" % 
            (np.mean([1 if score == 1 else 0 for score in test_text_scores]), np.mean(test_text_scores), np.mean(test_text_worse)))
        '''
        print(test_correct)
        print(test_q0_correct)
        print(test_q1_correct)
        print(test_q2_correct)
        print(test_oracle_correct)
        print(test_base_correct)
        print(test_score_correct)
        print(test_text_correct)
        '''
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type = str, default = 'MSDialog')
    parser.add_argument('--topn', type = int, default = 1)
    parser.add_argument('--cv', type = int, default = -1)
    parser.add_argument('--reranker_name', type = str, default = 'Poly')
    args = parser.parse_args()
    main(args)
