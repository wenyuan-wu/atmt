#!/usr/bin/env bash

dest_dir=output_bpe_dropout
mkdir -p $dest_dir

python translate.py --data bpe_dropout/prepared_data --checkpoint-path checkpoints_bpe_dropout/checkpoint_best.pt --output \
$dest_dir/model_translation.txt --cuda True

# restore translation from BPE
sed -r 's/(@@ )|(@@ ?$)//g' $dest_dir/model_translation.txt > $dest_dir/model_translation.out

# same post process as baseline
cat $dest_dir/model_translation.out | perl moses_scripts/detruecase.perl | \
perl moses_scripts/detokenizer.perl -q -l en > $dest_dir/translation.txt

cat $dest_dir/translation.txt | sacrebleu baseline/raw_data/test.en > $dest_dir/result.txt
echo "Translation and evaluation results saved in $dest_dir"
