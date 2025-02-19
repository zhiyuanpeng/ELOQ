from tqdm import tqdm,trange
import pytrec_eval
import numpy as np
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def add_instruct_concatenate(texts,instruction):
    return [instruction+t for t in texts]

def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_scores(query_ids,doc_ids,scores):
    assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0])==len(doc_ids),f"{len(scores[0])}, {len(doc_ids)}"
    emb_scores = {}
    for query_id,doc_scores in zip(query_ids,scores):
        cur_scores = {}
        for did,s in zip(doc_ids,doc_scores):
            cur_scores[str(did)] = s
        cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:1000]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores

def retrieval_bm25(queries,query_ids,documents,doc_ids,**kwargs):
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    corpus = [analyzer.analyze(x) for x in documents]
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                        normalize_queries=False, normalize_documents=False)
    all_scores = {}
    bar = tqdm(queries, desc="BM25 retrieval")
    for query_id, query in zip(query_ids, queries):
        bar.update(1)
        query = analyzer.analyze(query)
        bm25_query = model[dictionary.doc2bow(query)]
        similarities = bm25_index[bm25_query].tolist()
        all_scores[str(query_id)] = {}
        for did, s in zip(doc_ids, similarities):
            all_scores[str(query_id)][did] = s
        cur_scores = sorted(all_scores[str(query_id)].items(),key=lambda x:x[1],reverse=True)[:1000]
        all_scores[str(query_id)] = {}
        for pair in cur_scores:
            all_scores[str(query_id)][pair[0]] = pair[1]
    return all_scores

@torch.no_grad()
def retrieval_sbert_bge(queries,query_ids,documents,doc_ids,instructions,model_id,cache_dir,**kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_id=='bge':
        model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
        queries = add_instruct_concatenate(texts=queries,instruction=instructions['query'])
    elif model_id=='sbert':
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
    else:
        raise ValueError(f"The model {model_id} is not supported")
    batch_size = kwargs.get('batch_size',128)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, f"batchsize_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, f"batchsize_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, f"batchsize_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    query_emb = model.encode(queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores)

@torch.no_grad()
def retrieval_sf_qwen_e5(queries,query_ids,documents,doc_ids,model_id,instructions,cache_dir,**kwargs):
    if model_id=='sf':
        tokenizer = AutoTokenizer.from_pretrained('salesforce/sfr-embedding-mistral')
        model = AutoModel.from_pretrained('salesforce/sfr-embedding-mistral',device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    elif model_id=='qwen':
        tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', device_map="auto", trust_remote_code=True).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='qwen2':
        tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen2-7b-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen2-7b-instruct', device_map="auto", trust_remote_code=True).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='e5':
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    elif model_id=='linq':
        tokenizer = AutoTokenizer.from_pretrained('Linq-AI-Research/Linq-Embed-Mistral')
        model = AutoModel.from_pretrained('Linq-AI-Research/Linq-Embed-Mistral', device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)  
    elif model_id=='stella':
        tokenizer = AutoTokenizer.from_pretrained('NovaSearch/stella_en_1.5B_v5')
        model = AutoModel.from_pretrained('NovaSearch/stella_en_1.5B_v5', device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    else:
        raise ValueError(f"The model {model_id} is not supported")
    model = model.eval()
    queries = add_instruct_concatenate(texts=queries,instruction=instructions['query'])
    batch_size = kwargs.get('encode_batch_size',1)

    doc_emb = None
    cache_path = os.path.join(cache_dir, 'doc_emb', model_id, f"batchsize_{batch_size}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.isfile(cache_path):
        # already exists so we can just load it
        doc_emb = np.load(cache_path, allow_pickle=True)
    
    for start_idx in trange(0,len(documents),batch_size):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > start_idx:
            continue

        batch_dict = tokenizer(documents[start_idx:start_idx+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
        # doc_emb[start_idx] = embeddings
        doc_emb = embeddings if doc_emb is None else np.concatenate((doc_emb, np.array(embeddings)), axis=0)

        # save the embeddings every 1000 iters, you can adjust this as needed
        if (start_idx + 1) % 1000 == 0:
            np.save(cache_path, doc_emb)
        
    np.save(cache_path, doc_emb)

    doc_emb = torch.tensor(doc_emb)
    print("doc_emb shape:",doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    query_emb = []
    for start_idx in trange(0, len(queries), batch_size):
        batch_dict = tokenizer(queries[start_idx:start_idx + batch_size], max_length=max_length, padding=True,
                               truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().tolist()
        query_emb += embeddings
    query_emb = torch.tensor(query_emb)
    print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores)


RETRIEVAL_FUNCS = {
    'sf': retrieval_sf_qwen_e5,
    'qwen': retrieval_sf_qwen_e5,
    'qwen2': retrieval_sf_qwen_e5,
    'e5': retrieval_sf_qwen_e5,
    'stella': retrieval_sf_qwen_e5,
    'linq': retrieval_sf_qwen_e5,
    'bm25': retrieval_bm25,
    'sbert': retrieval_sbert_bge,
    'bge': retrieval_sbert_bge,
    # 'inst-l': retrieval_instructor,
    # 'inst-xl': retrieval_instructor,
    # 'grit': retrieval_grit,
    # 'cohere': retrieval_cohere,
    # 'voyage': retrieval_voyage,
    # 'openai': retrieval_openai,
    # 'google': retrieval_google
}

def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 50]):
    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    output = {
        **ndcg, 
        # **_map, 
        **recall, 
        # **precision, 
        **mrr}
    # print(output)
    return output
