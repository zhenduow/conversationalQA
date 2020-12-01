# conversational search
A risk-aware conversational search system consisting of pretrained answer and question rerankers and a decision maker trained by reinforcement learning.


* How to use
  * Preprocess data (MSDialog as example)
    ```
    cd data
    python3 process_MSDialog.py
    ```
    This will process and filter the data. All conversations that meet the filtering criterion are saved in MSDialog-Complete. The others are save in MSDialog-Incomplete. The former is used for the main experiments and the latter is used for fine-tuning the rerankers only.
  * Fine-tune pretrained reranker checkpoints on dataset (MSDialog as example)
    1. process the raw data into ParlAI readable training/test sets.
    '''
    
    '''
