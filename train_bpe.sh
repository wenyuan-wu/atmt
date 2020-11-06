#!/usr/bin/env bash

mkdir -p checkpoints_bpe

python train.py --data bpe/prepared_data --source-lang de --target-lang en --save-dir checkpoints_bpe \
--log-file checkpoints_bpe/train_log.txt --cuda True
echo "log file saved"

# test training on tiny data set
#python train.py --data bpe/prepared_data --source-lang de --target-lang en --save-dir checkpoints_bpe \
#--log-file checkpoints_bpe/train_log.txt --cuda True --train-on-tiny
#echo "log file saved"
