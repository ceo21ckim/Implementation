### FastText+LSTM
**Dataset**
'Amazon': [url](http://jmcauley.ucsd.edu/data/amazon/links.html)


**1.Tokenizer**

`google`에서 만든 `sentencepiece`를 사용하여 `BPE`를 통해 `tokenizer` 했습니다.
```
import sentencepiece as spm 

spm.SentencePieceTraniner.Train(
f'--input={input_txt} --model_prefix={saving_name} --vocab_size={vocab_size + 4} --model_type=bpe --max_sentence_length=9999' + 
'--pad_id=0 --pad_piece=[PAD]' + 
'--unk_id=1 --unk_piece=[UNK]' + 
'--bos_id=2 --bos_piece=[BOS]' + 
'--eos_id=3 --eos_piece=[EOS]'
```
  
  
**2.Word Embedding**

`FastText`를 활용하여 `word_embedding`을 수행한 후 `LSTM`의 입력으로 사용하였습니다.
```
from gensim.models import FastText

# vector_size = embedding size, (sg=1 -> skip-gram, sg=0 -> CBoW)
embedded = FastText(sentences, vector_size=128, window=10, min_count=2, sg=1)
```
