#!/usr/bin/env bash

mkdir -p checkpoints_bpe_dropout

python train_bpe_dropout.py --data bpe_dropout/prepared_data --source-lang de --target-lang en --save-dir checkpoints_bpe_dropout \
--log-file checkpoints_bpe_dropout/train_log.txt --cuda True
echo "log file saved"

# test training on tiny data set
#python train_bpe_dropout.py --data bpe_dropout/prepared_data --source-lang de --target-lang en --save-dir checkpoints_bpe_dropout \
#--log-file checkpoints_bpe_dropout/train_log.txt --cuda True --train-on-tiny
#echo "log file saved"
