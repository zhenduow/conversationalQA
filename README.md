# conversational search
A risk-aware conversational search system consisting of pretrained answer and question rerankers and a decision maker trained by reinforcement learning.


## How to use
1. Preprocess data. Here we use [MSDialog dataset](https://ciir.cs.umass.edu/downloads/msdialog/) as example. You can set dataset_name to be 'UDC' or 'Opendialkg' for [Ubuntu Dialog Corpus](http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/) or [Opendialkg](https://github.com/facebookresearch/opendialkg) currently.
    ```
    $ cd data
    $ python3 data_processing.py --dataset_name MSDialog
    ```
    This will process and filter the data. All conversations that meet the filtering criterion are saved in MSDialog-Complete and will be automatically split into training and testing set. The others are save in MSDialog-Incomplete. The former is used for the main experiments and the latter is used for fine-tuning the rerankers only.
1. Fine-tune pretrained reranker checkpoints on dataset (MSDialog as example)
    ```
    $ cd ParlAI
    $ python3.6 -u examples/train_model.py \
        --init-model zoo:pretrained_transformers/poly_model_huge_reddit/model \
        -t fromfile:parlaiformat --fromfile_datapath ../data/MSDialog-parlai-answer \
        --model transformer/polyencoder --batchsize 4 --eval-batchsize 100 \
        --warmup_updates 100 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 \
        -lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 \
        --text-truncate 360 --num-epochs 16.0 --max_train_time 200000 -veps 0.5 \
        -vme 8000 --validation-metric accuracy --validation-metric-mode max \
        --save-after-valid True --log_every_n_secs 20 --candidates batch --fp16 True \
        --dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 \
        --variant xlm --reduction-type mean --share-encoders False \
        --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 \
        --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 \
        --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 \
        --learn-embeddings True --polyencoder-type codes --poly-n-codes 64 \
        --poly-attention-type basic --dict-endtoken __start__ \
        --model-file zoo:pretrained_transformers/model_poly/answer \
        --ignore-bad-candidates True  --eval-candidates batch
    $ python3.6 -u examples/train_model.py \
        --init-model zoo:pretrained_transformers/poly_model_huge_reddit/model \
        -t fromfile:parlaiformat --fromfile_datapath ../data/MSDialog-parlai-question \
        --model transformer/polyencoder --batchsize 4 --eval-batchsize 100 \
        --warmup_updates 100 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 \
        -lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 \
        --text-truncate 360 --num-epochs 16.0 --max_train_time 200000 -veps 0.5 \
        -vme 8000 --validation-metric accuracy --validation-metric-mode max \
        --save-after-valid True --log_every_n_secs 20 --candidates batch --fp16 True \
        --dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 \
        --variant xlm --reduction-type mean --share-encoders False \
        --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 \
        --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 \
        --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 \
        --learn-embeddings True --polyencoder-type codes --poly-n-codes 64 \
        --poly-attention-type basic --dict-endtoken __start__ \
        --model-file zoo:pretrained_transformers/model_poly/question \
        --ignore-bad-candidates True  --eval-candidates batch
    ```
    This will download the poly-encoder checkpoints pretrained on reddit and fine-tune it on our preprocessed dataset. The fine-tuning code is based on [ParlAI poly-encoder](https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder/), but we modify several scripts for our needs.
