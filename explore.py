import string
from typing import List, Dict

from tinygrad.helpers import tqdm
from datasets import load_dataset


TOKENS: Dict[str, int] = {}         # tokenizer
SNEKOT: Dict[int, str] = {}         # de-tokenizer
TASK = "qa3" # ["qa1",...,"qa20"]
LANG = "en"


def preprocess(text: str) -> str:
  text = text.lower()               # turn lowercase
  text = text.strip()               # remove dangling whitespace
  for p in string.punctuation:      # remove punctuation
    text = text.replace(p, "")
  return text


def parse(data) -> (List[str], List[str]):
  _data = data["story"]
  inputs, targets = [], []
  context = ""
  for (text, answer, _type) in zip(_data["text"],
                                   _data["answer"],
                                   _data["type"]):
    if _type == 1:
      inputs.append(context + " " + text)
      targets.append(answer)
      context = ""
    else:
      context = context + " " + text if context != "" else text

  inputs = [preprocess(_) for _ in inputs]
  targets = [preprocess(_) for _ in targets]
  return inputs, targets


def tokenize(text: str) -> List[int]:
  ret = []
  for word in text.split():
    if word not in TOKENS.keys():
      new_token = len(TOKENS.keys())
      TOKENS[word] = new_token
      SNEKOT[new_token] = word
    ret.append(TOKENS[word])
  return ret

def detokenize(tokens: List[int]) -> str:
  words = [SNEKOT[_] for _ in tokens]
  return " ".join(words)

def pipeline(dataset, key) -> (List[int], List[int]):
  X, Y = [], []
  for data in dataset[key]:
    x, y = parse(data)
    X.extend(tokenize(x_i) for x_i in x)
    Y.extend(tokenize(y_i) for y_i in y)
  return X, Y

if __name__ == "__main__":
  print("=== loading, parsing, preprocessing and tokenizing dataset ===")
  dataset = load_dataset("babi_qa", type=LANG, task_no=TASK)
  X_train, Y_train = pipeline(dataset, "train")
  X_test,  Y_test  = pipeline(dataset, "test")

  print(detokenize(X_train[50]))
