"""
BigBird, is a sparse-attention to the input sequence. Theoretically, it has been shown that applying sparse, global, and random attention approximates full attention.
Current implementation supports only ITC, on the other side doesn`t support num_random_blocks=0

single sequence: [CLS] X [SEP]
pair of sequence: [CLS] A [SEP] B [SEP]

Created a mask from the two sequences passed to be used in a sequence-pair classification task, that following format:

0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 
|  first sequence   |  second sequence  |


"""
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/bigbird-roberta-base": "https://huggingface.co/google/bigbird-roberta-base/resolve/main/spiece.model",
        "google/bigbird-roberta-large": "https://huggingface.co/google/bigbird-roberta-large/resolve/main/spiece.model",
        "google/bigbird-base-trivia-itc": "https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/spiece.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/bigbird-roberta-base": 4096,
    "google/bigbird-roberta-large": 4096,
    "google/bigbird-base-trivia-itc": 4096,
}

"""
class BigBirdTokenizer(PreTrainedTokenizer):
    Construct a BigBird tokenizer. Based on SentencePiece

    This tokenizer inherits from 'PreTrainedTokenizer' which contains most of the main methods.

    Args:
        vocab_file: SentencePiece
        eos_token: </s>
        bos_token: <s>
        unk_token: <unk>
        pad_token: <pad>
        sep_token: [SEP]
        cls_token: [CLS]
        mask_token: [MASK]
        

class BigBirdModel(BigBirdPreTrainedModel):
    Args:
        input_ids: Indices of input sequence tokens in the vocabluary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        position_ids: Indices of positions of each input sequence tokens in the position embeddings.
        head_mask: Mask to nullify selected heads of the self-attention modules.
        inputs_embeds: (batch_size, sequence_length, hidden_size)
        encoder_hidden_states: Sequence of hidden-states at the output of the last layer of the encoder.
        encoder_attention_mask: Mask to avoid performing attention on the padding token indices of the encoder input.

    
    Returns:
        last_hidden_state: Sequence of hidden-states at the output fo the last layer of the model.
        pooler_output: Hast layer hidden-states of the first token of the sequence.
        hidden_states: one for the output of the embeddings.
        attentions
        cross_attentions
        past_key_values

"""

from transformers import BigBirdPreTrainedModel, BigBirdTokenizer, BigBirdModel 
import torch 

tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
model = BigBirdModel.from_pretrained('google/bigbird-roberta-base')

inputs = tokenizer("Hello, my dog is cute", return_tensors='pt')
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
outputs.__dict__.keys()

def example(sent = "Hello, my dog is cute"):

    inputs = tokenizer(sent, return_tensors='pt')
    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    return print(f'sentence: {sent} \ninputs: {inputs}, \noutputs: {outputs.__dict__.keys()}')

example()

########################################################################################################################

import argparse, os, warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='unstructure_dm_final_assignment')
parser.add_argument(
    '--batch_size', 
    default=64,
    type=int
)

parser.add_argument(
    '--example', 
    default=True,
    type=bool, 
    help='if want see example, input "True" in example parameter'
)

squad_v2 = False 

from datasets import load_dataset, load_metric 

d_sets = load_dataset('squad_v2' if squad_v2 else 'squad')

d_sets['train'][0]
"""
{'id': 
    '5733be284776f41900661182', 
'title': 
    'University_of_Notre_Dame', 
'context': 
    'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. 
    Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". 
    Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. 
    It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive 
    (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.', 

'question': 
    'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?', 
'answers': 
    {'text': 
        ['Saint Bernadette Soubirous'], 
    'answer_start': 
        [515]}}
"""

import datasets 
from datasets import ClassLabel, Sequence 
import random 
import pandas as pd 

def show_random_elements(dataset, num_examples=10):
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].tranform(lambda x: typ.names[x])
        
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.name[i] for i in x])

    return df

a = show_random_elements(d_sets['train'])

import transformers

a.context.apply(lambda x: x.__len__())
# 528, 539, 506, 1007, 1259, 767, 938, 869, 646, 179

train_set = d_sets['train']
valid_set = d_sets['validation']

valid_set = valid_set.filter(lambda x: len(x['context']) >0)



from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
import torch

model_id = "google/bigbird-base-trivia-itc"
model = BigBirdForQuestionAnswering.from_pretrained(model_id).to('cuda')
tokenizer = BigBirdTokenizer.from_pretrained(model_id)

assert isinstance(tokenizer, transformers.BigBirdForQuestionAnswering), 'please input BigBirdForQuestionAnswering'
tokenizer('The final assignment is to implemention of Bigbird.')


PUNCTUATION_SET_TO_EXCLUDE = set(''.join(['‘', '’', '´', '`', '.', ',', '-', '"']))


def get_sub_answers(answers, begin=0, end=None):
  return [" ".join(x.split(" ")[begin:end]) for x in answers if len(x.split(" ")) > 1]


def expand_to_aliases(given_answers, make_sub_answers=False):
  if make_sub_answers:
    given_answers = given_answers + get_sub_answers(given_answers, begin=1) + get_sub_answers(given_answers, end=-1)
  answers = []
  for answer in given_answers:
    alias = answer.replace('_', ' ').lower()
    alias = ''.join(c if c not in PUNCTUATION_SET_TO_EXCLUDE else ' ' for c in alias)
    answers.append(' '.join(alias.split()).strip())
  return set(answers)


def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = torch.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]



def evaluate(example):
    encoding = tokenizer(example["question"], example["context"], return_tensors="pt", max_length=4096, padding="max_length", truncation=True)
    input_ids = encoding.input_ids.to("cuda")

    with torch.no_grad():
        start_scores, end_scores = model(input_ids=input_ids).to_tuple()

    start_score, end_score = get_best_valid_start_end_idx(start_scores[0], end_scores[0], top_k=8, max_size=16)

    all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
    answer_tokens = all_tokens[start_score: end_score + 1]

    example["output"] = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

    answers = expand_to_aliases(example["answers"]['text'], make_sub_answers=True)
    predictions = expand_to_aliases([example["output"]])

    example["match"] = len(list(answers & predictions)) > 0

    return example

results_short = valid_set.map(evaluate)

print("Exact Match (EM): {:.2f}".format(100 * sum(results_short['match'])/len(results_short)))


wrong_results = results_short.filter(lambda x: x['match'] is False)
print(f"\nWrong examples: ")
print_out = wrong_results.map(lambda x, i: print(f"{i} - Output: {x['output']} - Target: {x['norm_target']}"), with_indices=True)

