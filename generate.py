import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    # tokenizer = get_kobart_tokenizer()
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()


while 1:
    text = input('News > ').strip()
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=20, num_beams=5, no_repeat_ngram_size=1)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Summary > {}".format(output))
