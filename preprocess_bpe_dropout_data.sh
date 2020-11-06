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

subword-nmt learn-joint-bpe-and-vocab --input train.de train.en -s 4000 -o joint_de_en.code --write-vocabulary vocab.de vocab.en

subword-nmt apply-bpe -c joint_de_en.code --vocabulary vocab.de --vocabulary-threshold 1 --dropout 0.1 --seed 1024 < train.de.bk > train.de.drop
subword-nmt apply-bpe -c joint_de_en.code --vocabulary vocab.de --vocabulary-threshold 1 < train.de.bk > train.de
subword-nmt apply-bpe -c joint_de_en.code --vocabulary vocab.en --vocabulary-threshold 1 < train.en.bk > train.en

subword-nmt apply-bpe -c joint_de_en.code --vocabulary vocab.de --vocabulary-threshold 1 < test.de.bk > test.de
subword-nmt apply-bpe -c joint_de_en.code --vocabulary vocab.en --vocabulary-threshold 1 < test.en.bk > test.en

subword-nmt apply-bpe -c joint_de_en.code --vocabulary vocab.de --vocabulary-threshold 1 < valid.de.bk > valid.de
subword-nmt apply-bpe -c joint_de_en.code --vocabulary vocab.en --vocabulary-threshold 1 < valid.en.bk > valid.en

subword-nmt apply-bpe -c joint_de_en.code --vocabulary vocab.de --vocabulary-threshold 1 < tiny_train.de.bk > tiny_train.de
subword-nmt apply-bpe -c joint_de_en.code --vocabulary vocab.en --vocabulary-threshold 1 < tiny_train.en.bk > tiny_train.en

cd ../..
mkdir -p bpe_dropout/prepared_data
python preprocess.py --target-lang en --source-lang de --dest-dir bpe_dropout/prepared_data/ --train-prefix bpe_dropout/preprocessed_data/train \
--valid-prefix bpe_dropout/preprocessed_data/valid --test-prefix bpe_dropout/preprocessed_data/test --tiny-train-prefix bpe_dropout/preprocessed_data/tiny_train \
--threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000
