#!/usr/bin/env bash

mkdir output_bpe

python translate.py --data baseline/prepared_data --checkpoint-path checkpoints_bpe/checkpoint_best.pt --output output_bpe/model_translation.txt

spm_decode --model=baseline/raw_data/spm.model < output_bpe/model_translation.txt > output_bpe/model_translation.out

cat output_bpe/model_translation.out | sacrebleu baseline/raw_data/test.en > output_bpe/result.txt

rm -r baseline/raw_data
mv baseline/raw_data_bak baseline/raw_data
