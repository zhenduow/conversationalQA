# conversational search
A risk-aware conversational search system consisting of pretrained answer and question rerankers and a decision maker trained by reinforcement learning.


## How to use
1. Preprocess data. Here we use [MSDialog dataset](https://ciir.cs.umass.edu/downloads/msdialog/) as example. You can set dataset_name to be 'UDC' or 'Opendialkg' for [Ubuntu Dialog Corpus](http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/) or [Opendialkg](https://github.com/facebookresearch/opendialkg) currently.
    ```
    cd data
    python3 data_processing.py --dataset_name MSDialog
    ```
    This will process and filter the data. All conversations that meet the filtering criterion are saved in MSDialog-Complete and will be automatically split into training and testing set. The others are save in MSDialog-Incomplete. The former is used for the main experiments and the latter is used for fine-tuning the rerankers only.
1. Fine-tune pretrained reranker checkpoints on dataset (MSDialog as example)
    1. 
