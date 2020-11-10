#!/usr/bin/env bash

mkdir -p bpe_dropout
cp -r baseline/preprocessed_data bpe_dropout/preprocessed_data
cd bpe_dropout/preprocessed_data || exit
cp train.de train.de.bk
cp train.en train.en.bk
cp test.de test.de.bk
cp test.en test.en.bk
cp valid.de valid.de.bk
cp valid.en valid.en.bk
cp tiny_train.de tiny_train.de.bk
cp tiny_train.en tiny_train.en.bk

# learn BPE codes and generate vocabulary files
subword-nmt learn-joint-bpe-and-vocab --input train.de train.en -s 4000 -o joint_de_en.code --write-vocabulary vocab.de vocab.en

# copy vocabulary files into prepared data for training
cd ..
mkdir -p prepared_data
cp preprocessed_data/vocab.de prepared_data/dict.de
cp preprocessed_data/vocab.en prepared_data/dict.en
