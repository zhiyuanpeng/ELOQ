import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
from utils import load_documents, load_queries, load_qrels
from retrievers import RETRIEVAL_FUNCS,calculate_retrieval_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,help="Path to the root directory of the data. \"ELOQ/data\"")
parser.add_argument('--model', type=str, required=True,
                    choices=['bm25','cohere','e5','google','grit','inst-l','inst-xl',
                            'openai','qwen','qwen2','sbert','sf','voyage','bge', 'stella', 'linq'])
parser.add_argument('--corpus', type=str, choices=['20','200','full'],help="Corpus size to use.")
parser.add_argument('--cache_dir', type=str, default='cache',help="Directory to store cached embeddings.")
parser.add_argument('--config_dir', type=str, default='configs',help="Directory to store model configurations.")
parser.add_argument('--output_dir', type=str, default='outputs',help="Directory to store output results.")
args = parser.parse_args()

# Create output directory if it does not exist
if args.output_dir == "outputs":
    args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir, args.corpus, args.model)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
# Create cache directory if it does not exist
if args.cache_dir == "cache":
    args.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.cache_dir, args.corpus, args.model)
if not os.path.exists(args.cache_dir):
    os.makedirs(args.cache_dir)

# load documents
doc_ids, documents = load_documents(args.data_dir, args.corpus)
# Load queries
query_ids, queries = load_queries(args.data_dir)
# Load qrels
qrels_path = os.path.join(args.data_dir, "ELOQ", "ELOQ", "silver.csv")
out_of_scope_gt, in_scope_gt = load_qrels(qrels_path)

# Run retrieval
score_file_path = os.path.join(args.output_dir,f'score.json')
if not os.path.isfile(score_file_path):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_dir,args.model,"config.json")) as f:
        config = json.load(f)
    kwargs = {}
    scores = RETRIEVAL_FUNCS[args.model](
        queries=queries, query_ids=query_ids, documents=documents,
        doc_ids=doc_ids, instructions=config['instructions'], 
        model_id=args.model, cache_dir=args.cache_dir, **kwargs
    )
    with open(score_file_path,'w') as f:
        json.dump(scores,f,indent=2)
else: 
    with open(score_file_path) as f:
        scores = json.load(f)
    print(score_file_path,'exists')


out_of_scope_results = calculate_retrieval_metrics(results=scores, qrels=out_of_scope_gt, k_values=[1, 5, 10, 50])
in_scope_results = calculate_retrieval_metrics(results=scores, qrels=in_scope_gt, k_values=[1, 5, 10, 50])

print(f"Out-of-scope Results: {out_of_scope_results}")
print(f"In-scope Results: {in_scope_results}")