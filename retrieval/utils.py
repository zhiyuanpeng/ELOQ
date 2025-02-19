import os
import json 
import csv
import pandas as pd

def load_documents(data_dir, corpus_size):
    """
    Load doc_ids and documents from the data directory.
    Curretly default as loading 200 docs from each topic folder.

    Args:
        data_dir (str): Path to the root data directory.
        corpus_size (str): Size of the corpus to load. Can be "20", "200", or "full"

    Returns:
        doc_ids (list): List of document IDs.
        documents (list): List of document texts.
    """
    doc_ids = []
    documents = []

    # Iterate over topic folders
    topic_dir = os.path.join(data_dir, "ELOQ", "News1k2024-300")
    for topic_folder in os.listdir(topic_dir):
        if corpus_size == "20":
            topic_path = os.path.join(topic_dir, topic_folder, "20", "docs_in.csv")
        elif corpus_size == "200":
            topic_path = os.path.join(topic_dir, topic_folder, "200", "docs_in.csv")
        elif corpus_size == "full":
            subdirs = os.listdir(os.path.join(topic_dir, topic_folder))
            # Find the directory that is neither "20" nor "200"
            full_path_dir = next((d for d in subdirs if d not in {"20", "200"}), None)
            topic_path = os.path.join(topic_dir, topic_folder, full_path_dir, "docs_in.csv")

        if os.path.exists(topic_path):
            df = pd.read_csv(topic_path)

            if {'doc_id', 'document'}.issubset(df.columns):
                # doc_ids.extend(df['doc_id'].tolist()) # if the doc_is is like "news_1"
                doc_ids.extend(df['doc_id'].astype(str).apply(lambda x: x.split(".")[0]).tolist()) # if the doc_id is like "news_1.json"
                documents.extend(df['document'].tolist())
        else:
            print(f"Folder {topic_folder} does not have a docs_in.csv file.")

    return doc_ids, documents

def load_queries(data_dir):
    """
    Load queries from questions.json inside the ELOQ directory.

    Args:
        data_dir (str): Path to the root data directory.

    Returns:
        query_ids (list): List of query IDs.
        queries (list): List of query texts.
    """
    queries_path = os.path.join(data_dir, "ELOQ", "ELOQ", "questions.json")

    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Questions file not found at {queries_path}")

    with open(queries_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    query_ids = list(questions.keys())
    queries = list(questions.values())

    return query_ids, queries

def load_qrels(qrels_path):
    out_of_scope_gt = {}
    in_scope_gt = {}
    
    with open(qrels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = row["q_id"].strip()
            doc_id = row["doc_id"].strip()
            label = row["llm_confusion_label"].strip().lower()
            
            if label == "yes":
                if q_id not in out_of_scope_gt:
                    out_of_scope_gt[q_id] = {}
                out_of_scope_gt[q_id][doc_id] = 1  # Binary relevance
            elif label == "no":
                if q_id not in in_scope_gt:
                    in_scope_gt[q_id] = {}
                in_scope_gt[q_id][doc_id] = 1  # Binary relevance
    
    return out_of_scope_gt, in_scope_gt