from settings import *
import sentencepiece as spm
import argparse, pickle

from gensim.models import FastText 

from tqdm import trange, tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    '--vocab_size', 
    '-v', 
    default=5000, 
    type=int, 
    help='vocab size'
)

parser.add_argument(
    '--embedding_size', 
    '-e', 
    default=128, 
    type=int, 
    help='embedding size'
)

args = parser.parse_args()


def spm_trainer(domain, vocab_size):
    spm.SentencePieceTrainer.Train(
        f'--input={os.path.join(TXT_DIR, domain)}.txt --model_prefix={os.path.join(SPM_PATH, domain)} --vocab_size={vocab_size + 4} --model_type=bpe --max_sentence_length=9999' +
        '--pad_id=0 --pad_piece=[PAD]' + 
        '--unk_id=1 --unk_piece=[UNK]' + 
        '--bos_id=2 --bos_piece=[BOS]' + 
        '--eos_id=3 --eos_piece=[EOS]' 
    )


def tokenizer(domain):
    vocab_file = os.path.join(SPM_PATH, f'{domain}.model')
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)

    with open(os.path.join(TXT_DIR, f'{domain}.txt')) as f:
        strings = f.readlines()

    token_sent = []
    with trange(len(strings), desc=f'{domain} tokenizing..') as tr:
        for i in tr:
            token_sent.append(vocab.encode_as_pieces(strings[i].strip()))

    return token_sent


def word_embedding(model, file_name, sentences, embedding_size, window, min_count, sg):
    embedding = model(sentences, vector_size = embedding_size, window=window, min_count=min_count, sg=sg)
    embedding.save(f'{os.path.join(EMD_PATH, file_name)}.model')
    return print('complete saving word embedding!')


if __name__ == '__main__':
    for domain in tqdm(DATA_NAMES, desc='building vocaburary..'):
        spm_trainer(domain=domain, vocab_size=args.vocab_size)

        tokenized_sentences = tokenizer(domain)

        word_embedding(model=FastText, file_name=domain, sentences=tokenized_sentences, embedding_size=args.embedding_size, window=10, min_count=2, sg=1)
