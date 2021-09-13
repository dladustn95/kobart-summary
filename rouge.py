import json
from rouge import Rouge

# Load some sentences
def read(fn):
    f = open(fn, 'r', encoding="utf-8-sig")
    lines = []
    for line in f:
        lines.append(line.strip())

    f.close()

    return lines

hyp_path = read('logs/model_chp3/epoch=00.ckpt_result.txt')
ref_path = read('data/test.tgt')
rouge = Rouge()
scores = rouge.get_scores(hyps, refs, avg=True)
