import datasets 
import torch

validation_dataset = datasets.load_dataset("trivia_qa", 'rc', split="validation[:10%]", script_version='master')  # remove [:5%] to run on full validation set

validation_dataset.info.features

def format_dataset(example):
    example["context"] = " ".join(("\n".join(example["entity_pages"]["wiki_context"])).split("\n"))
    example["targets"] = example["answer"]["aliases"]
    example["norm_target"] = example["answer"]["normalized_value"]
    return example


validation_dataset = validation_dataset.map(format_dataset, remove_columns=["search_results", "question_source", "entity_pages", "answer", "question_id"])

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
    return df 

show_random_elements(validation_dataset, num_examples=3)


validation_dataset = validation_dataset.filter(lambda x: len(x["context"]) > 0)
# check out how many samples are left
validation_dataset

short_validation_dataset = validation_dataset.filter(lambda x: (len(x['question']) + len(x['context'])) < 4 * 4096)
short_validation_dataset


from transformers import BigBirdTokenizer, BigBirdForQuestionAnswering

tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-base-trivia-itc")
model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-base-trivia-itc").to("cuda")

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

    answers = expand_to_aliases(example["targets"], make_sub_answers=True)
    predictions = expand_to_aliases([example["output"]])

    example["match"] = len(list(answers & predictions)) > 0

    return example

results_short = short_validation_dataset.map(evaluate)

print("Exact Match (EM): {:.2f}".format(100 * sum(results_short['match'])/len(results_short)))
# 82.68 -> question + context


wrong_results = results_short.filter(lambda x: x['match'] is False)
print(f"\nWrong examples: ")
print_out = wrong_results.map(lambda x, i: print(f"{i} - Output: {x['output']} - Target: {x['norm_target']}"), with_indices=True)


print("Exact Match (EM): {:.2f}".format(100 * sum(results_short['match'])/len(results_short)))
# 62.15 / all token