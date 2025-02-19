import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import subprocess

def seed_everything(seed: int = None):
    print(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("set torch.backends.cudnn.benchmark=False")
    torch.backends.cudnn.benchmark = False
    print("set torch.backends.cudnn.deterministic=True")
    torch.backends.cudnn.deterministic = True


class LlamaResponseExtractor:
    def __init__(self, model_name=None, token=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=token).cuda()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
    
    def get_llama_response(self, input_ids):
        _, input_len = input_ids.size()
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=30)
        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return 1 if "yes" in response.lower() else 0

class LlamaPipline:
    def __init__(self, model_name=None, max_new_tokens=30):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.pipeline = pipeline("text-generation", model=model_name, tokenizer=model_name, device=torch.cuda.current_device())
    
    def get_llama_response(self, messages):
        response = self.pipeline(messages, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)[0]["generated_text"][-1]["content"].replace("\n", " ")
        return 1 if "yes" in response.lower() else 0, response
    
def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_trainable_params}')

def get_current_commit_id():
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_id
    except subprocess.CalledProcessError:
        return None

