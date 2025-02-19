import pandas as pd
import os
from tqdm import tqdm
from os.path import join, dirname
from dotenv import load_dotenv
load_dotenv()
import eloq.src.utils.doc_util as utils 
import eloq.src.utils.prompt_util as prompt_util
from eloq.src.utils.doc_util import doc_csv_schema, qrc_csv_schema, doc_files, qrc_files

def generate_RAG_responses(llm, doc_schema, doc_path, qr_schema, qr_path, method):
    df_in = utils.read_csv(doc_path, "Read the document-and-questions table from CSV file")
    print("Generate RAG response for each question, both original and confusing")
    rows_out = []
    for _, row in tqdm(df_in.iterrows(), total = df_in.shape[0]):
        doc_id = row[doc_schema["doc_id"]]
        document = row[doc_schema["document"]]
        orig_questions = utils.parse_numbered_questions(row[doc_schema["orig_qs"]])
        conf_questions = utils.parse_numbered_questions(row[doc_schema["conf_qs"]])
        qs = (
            [(q_id, "no" , q) for q_id, q in enumerate(orig_questions, start = 1)] +
            [(q_id, "yes", q) for q_id, q in enumerate(conf_questions, start = 1)]
        )
        for q_id, is_conf, q in qs:
            # F.6 RAG Basic
            if method == "Basic":
                response_q = prompt_util.generate_response(llm, document, q)
            # F.7 RAG Two-shot
            elif method == "Two-shot":
                response_q = prompt_util.generate_response_2shot(llm, document, q)
            # F.8 RAG Zero-shot-CoT
            elif method == "Zero-shot-CoT":
                response_q = prompt_util.generate_response_cot(llm, document, q)
            else:
                raise ValueError(f"Unknown method: {method}")
            row_out = {
                qr_schema["doc_id"] : doc_id,
                qr_schema["q_id"] : q_id,
                qr_schema["is_conf"] : is_conf,
                qr_schema["question"] : q,
                qr_schema["LLM_r"] : llm,
                qr_schema["response"] : response_q
            }
            rows_out.append(row_out)
        # break
    df_out = pd.DataFrame.from_dict(rows_out, dtype = str)
    utils.write_csv(df_out, qr_path, "Write the question-response table to CSV file")

if __name__ == "__main__":

    # Note: LLM_q may be stronger than LLM_r, to create more challenging confusions.

    llm_q = "gpt-4o-mini"  # LLM for generating questions (stronger)
    llm_eval = "gpt-4o-mini"
    news_num = 200
    doc_prompt = "dt-z-1"

    num_q_orig =  5  # Number of questions per original document
    num_fact = 6  # Number of questions per expanded document
    topics = [
        'sport', 'business', 'science', 'food',
        'politics', 'travel', 'entertainment',
        'music', 'news', 'tech'
    ]
    llm_models = ["gpt-3.5", "Meta-Llama-3.1-8B-Instruct-Turbo", "Meta-Llama-3.1-70B-Instruct-Turbo", "Llama-3.2-3B-Instruct-Turbo", "Mistral-7B-Instruct-v0.3", "Meta-Llama-3.3-70B-Instruct-Turbo"]
    affix = {"Basic": "", "Two-shot": "-twoshot", "Zero-shot-CoT": "-cot"}
    for llm_r in llm_models:
        for method in ["Basic", "Two-shot", "Zero-shot-CoT"]:
            # Skip gpt-3.5 for Basic as the responses are collected when running llm_question
            if llm_r == "gpt-3.5" and method == "Basic":
                continue
            tpoic_start_time = utils.get_time()
            for topic in topics:
                llm_start_time = utils.get_time()
                data_folder = f"data/processed/News1k2024-300/{topic}/{news_num}"
                exp_folder = f"data/experiments/llmq-{llm_q}/docp-{doc_prompt}/{news_num}/{topic}/llmr-{llm_r}" + affix[method]
                os.makedirs(exp_folder, exist_ok = True)
                doc_paths = {k : join(data_folder, v) if k == "in" else join(dirname(exp_folder), v)  for k, v in doc_files.items()}
                qrc_paths = {k : join(exp_folder, v) for k, v in qrc_files.items()}
                prompt_util.read_prompts(join("eloq/prompts"))

                print(f"\nCollect the responses of {llm_r} for {topic}\n")

                generate_RAG_responses(llm_r, doc_csv_schema, doc_paths["out"], qrc_csv_schema, qrc_paths[1], method)
                end_time = utils.get_time()
                print(f"{topic} {llm_r} takes: {end_time - llm_start_time:.2f} seconds")
            end_time = utils.get_time()
            print(f"{topic} takes: {end_time - tpoic_start_time:.2f} seconds")