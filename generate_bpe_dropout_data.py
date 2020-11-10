from subword_nmt import apply_bpe
import preprocess
import os
from seq2seq.data.dictionary import Dictionary


def process_bpe_dropout(code, vocab, in_name, out_name, dropout=0.0):
    """
    To apply BPE on desired data and output processed files.
    """
    codes = open(code, encoding='utf-8')
    vocab_file = open(vocab, encoding='utf-8')
    vocabulary = apply_bpe.read_vocabulary(vocab_file, 1)
    num_workers = apply_bpe.cpu_count()
    output_file = open(out_name, 'w', encoding='utf-8')
    bpe = apply_bpe.BPE(codes=codes, vocab=vocabulary)
    bpe.process_lines(in_name, output_file, dropout=dropout, num_workers=num_workers)


def preprocess_data(dest_dir,
                    train_prefix, tiny_train_prefix, valid_prefix, test_prefix,
                    source_lang, threshold_src, num_words_src, vocab_src,
                    target_lang, threshold_tgt, num_words_tgt, vocab_tgt,
                    ):
    """
    Take BPE data with dropout as input and translate it into binary files.
    """
    os.makedirs(dest_dir, exist_ok=True)
    src_dict = Dictionary.load(vocab_src)
    print('Loaded a source dictionary ({}) with {} words'.format(source_lang, len(src_dict)))
    # load existed vocabulary files instead of generating new one
    # src_dict = preprocess.build_dictionary([train_prefix + '.' + source_lang])
    # src_dict.finalize(threshold=threshold_src, num_words=num_words_src)
    # src_dict.save(os.path.join(dest_dir, 'dict.' + source_lang))
    # print('Built a source dictionary ({}) with {} words'.format(source_lang, len(src_dict)))

    tgt_dict = Dictionary.load(vocab_tgt)
    print('Loaded a target dictionary ({}) with {} words'.format(target_lang, len(tgt_dict)))
    # tgt_dict = preprocess.build_dictionary([train_prefix + '.' + target_lang])
    # tgt_dict.finalize(threshold=threshold_tgt, num_words=num_words_tgt)
    # tgt_dict.save(os.path.join(dest_dir, 'dict.' + target_lang))
    # print('Built a target dictionary ({}) with {} words'.format(target_lang, len(tgt_dict)))

    def make_split_datasets(lang, dictionary):
        if train_prefix is not None:
            preprocess.make_binary_dataset(train_prefix + '.' + lang, os.path.join(dest_dir, 'train.' + lang),
                                           dictionary)
        if tiny_train_prefix is not None:
            preprocess.make_binary_dataset(tiny_train_prefix + '.' + lang, os.path.join(dest_dir, 'tiny_train.' + lang),
                                           dictionary)
        if valid_prefix is not None:
            preprocess.make_binary_dataset(valid_prefix + '.' + lang, os.path.join(dest_dir, 'valid.' + lang),
                                           dictionary)
        if test_prefix is not None:
            preprocess.make_binary_dataset(test_prefix + '.' + lang, os.path.join(dest_dir, 'test.' + lang), dictionary)

    make_split_datasets(source_lang, src_dict)
    make_split_datasets(target_lang, tgt_dict)


def generate_bpe_dropout_data():
    """
    Function to pass desired arguments (files paths) to functions above to prepare data files for each training epoch.
    """
    code = 'bpe_dropout/preprocessed_data/joint_de_en.code'
    # apply BPE dropout on training data
    for prefix in ['train', 'tiny_train']:
        vocab = 'bpe_dropout/preprocessed_data/vocab.de'
        in_name = f'bpe_dropout/preprocessed_data/{prefix}.de.bk'
        out_name = f'bpe_dropout/preprocessed_data/{prefix}.de'
        process_bpe_dropout(code, vocab, in_name, out_name, dropout=0.1)

        vocab = 'bpe_dropout/preprocessed_data/vocab.en'
        in_name = f'bpe_dropout/preprocessed_data/{prefix}.en.bk'
        out_name = f'bpe_dropout/preprocessed_data/{prefix}.en'
        process_bpe_dropout(code, vocab, in_name, out_name, dropout=0.1)

    # no BPE dropout on validation and test data
    for prefix in ['valid', 'test']:
        vocab = 'bpe_dropout/preprocessed_data/vocab.de'
        in_name = f'bpe_dropout/preprocessed_data/{prefix}.de.bk'
        out_name = f'bpe_dropout/preprocessed_data/{prefix}.de'
        process_bpe_dropout(code, vocab, in_name, out_name)

        vocab = 'bpe_dropout/preprocessed_data/vocab.en'
        in_name = f'bpe_dropout/preprocessed_data/{prefix}.en.bk'
        out_name = f'bpe_dropout/preprocessed_data/{prefix}.en'
        process_bpe_dropout(code, vocab, in_name, out_name)

    dest_dir = 'bpe_dropout/prepared_data/'
    train_prefix = 'bpe_dropout/preprocessed_data/train'
    tiny_train_prefix = 'bpe_dropout/preprocessed_data/tiny_train'
    valid_prefix = 'bpe_dropout/preprocessed_data/valid'
    test_prefix = 'bpe_dropout/preprocessed_data/test'
    source_lang = 'de'
    vocab_src = 'bpe/prepared_data/dict.de'
    threshold_src = 1
    num_words_src = 4000
    target_lang = 'en'
    vocab_tgt = 'bpe/prepared_data/dict.en'
    threshold_tgt = 1
    num_words_tgt = 4000

    preprocess_data(dest_dir,
                    train_prefix, tiny_train_prefix, valid_prefix, test_prefix,
                    source_lang, threshold_src, num_words_src, vocab_src,
                    target_lang, threshold_tgt, num_words_tgt, vocab_tgt,
                    )


if __name__ == '__main__':
    generate_bpe_dropout_data()
