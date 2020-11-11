import logging
import sys
import numpy as np
from sklearn.metrics import ndcg_score

class User:
    '''
    A user simulator class that is used to simulate user in response to agent's questions.
    The user class is essentially a database of conversations.
    It search for the answer for each input question and return it if it finds,
    otherwise it count it as a bad question.
    After the bad question count reaches the patience threshold,
    the user will quit and send back a signal of that.
    '''
    def __init__(self, dataset, cq_reward, cq_penalty, tolerance = 2, patience = 5):
        '''
        :param dataset: (class) The dataset. The default MSDialog-complete dataset format is described in 
                        https://ciir.cs.umass.edu/downloads/msdialog/ 
                        The dataset should consist of a dictionary with the conversation identifiers as keys,
                        and lists of utterances as values.
        :param tolerance: (int) The maximum time that user tolerates a bad question from the agent 
        :param patience: (int) The maximum time that the user is willing to answer agent's question 
        :param anger: (int) The total count of bad question asked by agent. Initialize this as 0.
        :return: self
        '''
        self.dataset = dataset
        self.tolerance = tolerance
        self.patience = patience
        self.anger = 0
        self.cq_reward = cq_reward
        self.cq_penalty = cq_penalty

    def respond_to_question(self, conversation_id, question):
        '''
        User simulator responds to a question within the context of a given conversation.
        :param conversation_id: (str) The conversation identifier.
        :param question: (str) The question that needs response from the user simulator.
        :return: (str) The user simulator's response;
                0, if the question is not in the given conversation, i.e., a bad question.
                -1, if the question is in the given conversation but not followed up with response in the database. In this case, the question maybe an answer.
        '''
        is_off_topic = True
        question_pos = -1
        for pos, utterance in enumerate(self.dataset[conversation_id]):
            if question == utterance:
                is_off_topic = False
                question_pos = pos
        
        if is_off_topic:
            return 0
        else:
            try:
                return self.dataset[conversation_id][question_pos + 1]
            except:
                #logging.info('The question is the last utterance in the conversation.')
                return -1
    
    def simulate(self, conversation_id, agent):
        '''
        Simulate a whole conversation. The user simulator iteratively respond to the agent's input.
        The user simulator will quit the conversation if:
        1. The agent asks too many questions before retrieving answer. (more than self.patience)
        2. The agent asks too many bad questions before retrieving answer. (more than self.tolerance)

        :return: -1, if the conversation_id is not in the database.
        '''
        
        question_count = 0
        bad_question_count = 0
        logging.info('Simulating conversation ' + conversation_id)
        try:
            self.dataset[conversation_id]
        except:
            logging.info('The conversation ' + conversation_id + ' is not in database.')
            return -1

        query = self.dataset[conversation_id][0] # initialize the query with the first user post.

        while True:
            # system process the conversation and generate answer or question.
            is_answer = False # whether the agent returns an answer, set to True if the agent retrieves answer

            agent_utterance = agent.generate_question(query)
            logging.info('< ' + agent_utterance)

            if not is_answer:
                question_count += 1
                if question_count > self.patience:
                    query = query + ' ' + agent_utterance + 'I have provided enough information. Can you give me the answer immediately?'
                    continue
                cur_response = self.respond_to_question(conversation_id, agent_utterance)
                if cur_response == -1:
                    logging.info('The conversation ends here in the database. Or the clarifying question being asked is actually the final answer to the query')
                    break
                elif cur_response == 0:
                    bad_question_count += 1
                    if bad_question_count > self.tolerance:
                        query = query + ' ' + agent_utterance + 'That is not relevant to my problem.'
                        break
                else:
                    logging.info('> ' + cur_response)
                    query = query + ' ' + agent_utterance + ' ' + cur_response # update query status

            elif is_answer == '1':
                logging.info('Answer received. Evaluating.')
                pass


    def initialize_state(self, conversation_id):
        '''
        Initialize the user state given the conversation id.
        '''            
        initial_query = self.dataset[conversation_id][0]
        initial_query = initial_query[:-1] if initial_query[-1] == '\n' else initial_query
        self.anger = 0
        return initial_query
        

    def initialize_pretrain_state(self, conversation_id):
        '''
        Initialize the user state given the conversation id.
        '''
        final_query = self.dataset[conversation_id][0][:-1] if self.dataset[conversation_id][0][-1] == '\n' else self.dataset[conversation_id][0]    
        for ut in self.dataset[conversation_id][1:-1]:
            final_query += ' [SEP] '
            final_query += ut[:-1] if ut[-1] == '\n' else ut
        return final_query
    
    def update_state(self, conversation_id, context, action, top_n_question, top_n_answer, use_top_k):
        '''
        Read the agent action and update the user state, compute reward and return them for save.
        The agent action should be 0 (retrieve an answer) or 1 (ask clarifying question)
        '''
        if action == 0:
            # agent answer the question, evaluate the answer
            n = len(top_n_answer)
            context_ = context + ' [SEP] ' + top_n_answer[0]
            rel = [1]+[0]*(n-1)
            true_rel = [0] * n
            for i in range(n):
                try:
                    true_rel[i] = - self.respond_to_question(conversation_id, top_n_answer[i])
                except:
                    print("Error in conversation: " + conversation_id)
            reward = sum([(n*n)/(id+1) for id,n in enumerate(true_rel)])
            if reward > 1:
                reward = 1
            return context_, reward, True, None
        elif action == 1:
            # agent asks clarifying question, find corresponding answer in the dataset and return
            done = False
            correct_question_id = -1
            user_response_text = ''
            for qid in range(len(top_n_question)):
                response = self.respond_to_question(conversation_id, top_n_question[qid])
                if type(response) == int:
                    continue
                else:
                    if correct_question_id == -1:
                        #logging.info("Good CQ.")
                        correct_question_id = qid
                        user_response_text = response
            if 0 <= correct_question_id <= (use_top_k - 1):
                reward = self.cq_reward
                context_ = context + ' [SEP] ' + top_n_question[correct_question_id] + ' [SEP] ' + user_response_text
            else:
                # the agent asks a bad question  
                reward = self.cq_penalty
                done = True
                context_ = context + ' [SEP] ' + top_n_question[0] + ' [SEP] ' + 'This question is not relevant.'
            return context_, reward, done, top_n_question[correct_question_id]
