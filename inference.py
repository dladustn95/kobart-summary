import argparse
from train import KoBARTConditionalGeneration
from transformers.models.bart import BartForConditionalGeneration
import torch
from kobart import get_kobart_tokenizer
from tqdm import tqdm
import yaml

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    # tokenizer = get_kobart_tokenizer()
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, type=str)
parser.add_argument("--model_binary", default=None, type=str)
parser.add_argument("--testfile", default=None, type=str)
parser.add_argument("--outputfile", default=None, type=str)
args = parser.parse_args()


with open(args.hparams) as f:
    hparams = yaml.load(f)

inf = KoBARTConditionalGeneration.load_from_checkpoint(args.model_binary, hparams=hparams)
model = inf.model
model.to('cuda')
model.eval()
tokenizer = get_kobart_tokenizer()

lines = []
f = open(args.testfile, 'r', encoding="utf-8-sig")
for line in f:
    lines.append(line.strip())
f.close()

f = open(args.outputfile, 'w', encoding="utf-8-sig")
for line in tqdm(lines):
    input_ids = tokenizer.encode(line)
    if len(input_ids) > 512:
        input_ids = input_ids[:512] + [input_ids[-1]]
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.cuda() 
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5, no_repeat_ngram_size=1)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    f.write(output)
    f.write("\n")
f.close()
