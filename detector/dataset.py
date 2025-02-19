import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel 
import random
import json
from tqdm import tqdm 

class BaseTokenEmbeddingDataset(Dataset):
    def __init__(self, queries_path, corpus_path, tsv_path, tokenizer, llm, llm_name, prompt="Can this document answer the question? \nDocument: ", connection="\nQuestion: "):
        self.queries = self._load_json(queries_path)
        self.corpus = self._load_json(corpus_path)
        # Load (query-id, corpus-id, score) pairs
        if "golden" in tsv_path:
            self.data = self._load_gold_tsv(tsv_path)
        else:
            self.data = self._load_tsv(tsv_path)
            
        self.tokenizer = tokenizer
        self.llm = llm
        self.llm_name = llm_name
        self.prompt = prompt
        self.connection = connection
        self.special_id = 0 # Placeholder for the special token ID
        
    def _load_jsonl(self, path):
        data = {}
        with open(path, 'r') as f:
            for line in tqdm(f):
                item = json.loads(line.strip())
                data[item['_id']] = item['text']
        return data
    
    def _load_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def _load_tsv(self, path):
        data = []
        with open(path, 'r') as f:
            i = 0
            for line in tqdm(f):
                if i == 0:
                    i += 1
                    continue
                corpus_id, query_id, score = line.strip().split(',')
                score = 1.0 if score == "yes" else 0.0
                data.append((query_id, corpus_id, score))
        return data
    
    def _load_gold_tsv(self, path):
        data = []
        with open(path, 'r') as f:
            i = 0
            for line in tqdm(f):
                if i == 0:
                    i += 1
                    continue
                corpus_id, query_id, _, score, _, _ = line.strip().split(',')
                score = 1.0 if score == "yes" else 0.0
                data.append((query_id, corpus_id, score))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query_id, corpus_id, score = self.data[idx]
        query = self.queries[query_id]
        document = self.corpus[corpus_id]['title'] + " " + self.corpus[corpus_id]['content']
        return query, document, score, query_id, corpus_id

    def collate_fn(self, batch):
        tokenized_data = []
        all_query_ids, all_document_ids, all_scores = [], [], []

        for query, doc, score, query_id, doc_id in batch:
            # Create the prompt for each sample
            input_text = self.prompt + doc + self.connection + query + "\nAnswer: "
            
            if self.llm_name == "openai-community/gpt2": 
                tokenized = self.tokenizer(
                    input_text,
                    max_length=1024 - 1,  # Reserve space for special token, This limitation is only for GPT2 because it has context window of 1024 tokens
                    truncation=True,
                    add_special_tokens=False,
                    return_tensors="pt"
                )
            else: 
                tokenized = self.tokenizer(
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                
            input_ids = torch.cat([tokenized.input_ids, torch.tensor([[self.special_id]])], dim=1)
            attention_mask = torch.ones_like(input_ids)
            # 1 means this appended last token will participate in the attention mechanism
            attention_mask[:, -1] = 1
            
            tokenized_data.append({
                "input_ids": input_ids.squeeze(0),
                "attention_mask": attention_mask.squeeze(0)
            })
            
            all_query_ids.append(query_id)
            all_document_ids.append(doc_id)
            all_scores.append(score)

        padded_data = self.tokenizer.pad(
            tokenized_data,
            padding=True,
            return_tensors="pt"
        )
        input_ids = padded_data['input_ids'].cuda().long()
        attention_mask = padded_data['attention_mask'].cuda().long()
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.hidden_states[-1] # Shape: (batch_size, seq_length, hidden_size)

        # Extract hidden state of the special token from the last layer
        token_embeddings = []
        for i, input_id in enumerate(input_ids):
            # Find the position of the first special token in the sequence
            position = (input_id == self.special_id).nonzero(as_tuple=True)[0][0].item()
            token_embedding = hidden_states[i, position, :].squeeze(0) # Shape: (hidden_size,)
            token_embeddings.append(token_embedding)
        
        batch_of_token_embeddings = torch.stack(token_embeddings, dim=0) # Shape: (batch, hidden_size)

        return batch_of_token_embeddings, (all_query_ids, all_document_ids, all_scores)

class EOSTokenEmbeddingDataset(BaseTokenEmbeddingDataset):
    def __init__(self, queries_path, corpus_path, tsv_path, tokenizer, llm, llm_name, prompt="Can this document answer the question? \nDocument: ", connection="\nQuestion: "):
        super().__init__(queries_path, corpus_path, tsv_path, tokenizer, llm, llm_name, prompt, connection)
        self.special_id = self.tokenizer.eos_token_id # "2": </s> for Mistral. "128001": <|end_of_text|> for Llama. "50256": </s> for GPT2

class UnusedTokenEmbeddingDataset(BaseTokenEmbeddingDataset):
    def __init__(self, queries_path, corpus_path, tsv_path, tokenizer, llm, llm_name, prompt="Can this document answer the question? \nDocument: ", connection="\nQuestion: "):
        super().__init__(queries_path, corpus_path, tsv_path, tokenizer, llm, llm_name, prompt, connection)
        if llm_name == "mistralai/Mistral-7B-Instruct-v0.3":
            self.special_id = 557      # [control_555]
        elif llm_name == "meta-llama/Llama-3.2-3B-Instruct" or llm_name == "meta-llama/Llama-3.2-1B-Instruct" or llm_name == "meta-llama/Llama-3.1-8B-Instruct":
            self.special_id = 128002   # <|reserved_special_token_0|>
        elif llm_name == "openai-community/gpt2": 
            self.special_id = self.tokenizer.unk_token_id 

class SentenceBERTNLIFeatureDataset(BaseTokenEmbeddingDataset):
    def __init__(self, queries_path, corpus_path, tsv_path, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.queries = self._load_json(queries_path)
        self.corpus = self._load_json(corpus_path)
        # Load (query-id, corpus-id, score) pairs
        if "golden" in tsv_path:
            self.data = self._load_gold_tsv(tsv_path)
        else:
            self.data = self._load_tsv(tsv_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.cuda()
        self.model.eval()

    def collate_fn(self, batch):
        # [u; v; |u - v|] Embedding Implementation
        docs = [doc for _, doc, _, _, _ in batch]
        queries = [query for query, _, _, _, _ in batch]
        all_query_ids = [q_id for _, _, _, q_id, _ in batch]
        all_document_ids = [d_id for _, _, _, _, d_id in batch]
        all_scores = [score for _, _, score, _, _ in batch]
        
        # Tokenize documents
        doc_inputs = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        doc_inputs = {key: value.cuda() for key, value in doc_inputs.items()}
        doc_outputs = self.model(**doc_inputs)
        doc_cls = doc_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        
        # Tokenize queries
        query_inputs = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        query_inputs = {key: value.cuda() for key, value in query_inputs.items()}
        query_outputs = self.model(**query_inputs)
        query_cls = query_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

        # For NLI-based combination:
        # Construct the combined feature vector: [u; v; |u - v|]
        # where u = doc_cls and v = query_cls
        abs_diff = (doc_cls - query_cls).abs()
        combined_embeddings = torch.cat([doc_cls, query_cls, abs_diff], dim=1)  # Shape: (batch_size, 3 * hidden_size)

        return combined_embeddings, (all_query_ids, all_document_ids, all_scores)

        