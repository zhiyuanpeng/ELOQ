import pandas as pd
import os
from tqdm import tqdm
from os.path import join, dirname
from dotenv import load_dotenv
load_dotenv()
import eloq.src.utils.prompt_util as prompt_util
import eloq.src.utils.doc_util as utils
from eloq.src.utils.doc_util import doc_csv_schema, qrc_csv_schema, doc_files, qrc_files

def create_dictionary_of_indexed_documents(doc_schema, doc_path):
    df_doc = utils.read_csv(doc_path, "Read the document table from CSV file")
    print("Create a dictionary of indexed documents")
    documents = {
        row[doc_schema["doc_id"]] : row[doc_schema["document"]]
            for _, row in tqdm(df_doc.iterrows(), total = df_doc.shape[0])
    }
    return documents

def check_if_response_defused_confusion(llm, doc_schema, doc_path, qr_schema, qr_path_in, qr_path_out, n=1, shot=2):
    documents = create_dictionary_of_indexed_documents(doc_schema, doc_path)
    df_qr = utils.read_csv(qr_path_in, "Read the question-response table from CSV file")
    print("Ask LLM to check if its own response defused the confusion")
    rows_out = []
    for _, row in tqdm(df_qr.iterrows(), total = df_qr.shape[0]):
        doc_id = row[qr_schema["doc_id"]]
        document = documents[doc_id]
        question = row[qr_schema["question"]]
        # llm = row[qr_schema["LLM_r"]]
        response = row[qr_schema["response"]]
        # confusion = row[qr_schema["confusion"]]
        confusion = row[qr_schema["is_conf"]]
        if confusion == "no":
            defusion, is_defused = "n/a", "n/a"
        else:
            defusion, is_defused = prompt_util.check_response_for_defusion(llm, document, question, response, n, shot)
        row_out = dict(row)
        row_out[qr_schema["defusion"]] = defusion
        row_out[qr_schema["is_defused"]] = is_defused
        rows_out.append(row_out)
    df_out = pd.DataFrame.from_dict(rows_out, dtype = str)
    utils.write_csv(df_out, qr_path_out, "Write the question-response table to CSV file")

if __name__ == "__main__":
    # Note: LLM_q may be stronger than LLM_r, to create more challenging confusions.

    llm_q = "gpt-4o-mini"  # LLM for generating questions (stronger)
    # llm_r = "gpt-3.5"  # LLM for generating responses (weaker)
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
    llm_models = ["gpt-3.5", "Meta-Llama-3.1-8B-Instruct-Turbo", "Meta-Llama-3.1-70B-Instruct-Turbo", "Meta-Llama-3.3-70B-Instruct-Turbo", "Llama-3.2-3B-Instruct-Turbo", "Mistral-7B-Instruct-v0.3"]
    for llm_r in llm_models:
        for suffix in ["", "twoshot", "cot"]:
            tpoic_start_time = utils.get_time()
            for topic in topics:
                llm_start_time = utils.get_time()
                data_folder = f"data/processed/News1k2024-300/{topic}/{news_num}"
                if suffix:
                    exp_folder = f"data/experiments/llmq-{llm_q}/docp-{doc_prompt}/{news_num}/{topic}/llmr-{llm_r}-{suffix}"
                else:
                    exp_folder = f"data/experiments/llmq-{llm_q}/docp-{doc_prompt}/{news_num}/{topic}/llmr-{llm_r}"
                os.makedirs(exp_folder, exist_ok = True)

                doc_paths = {k : join(data_folder, v) if k == "in" else join(dirname(exp_folder), v)  for k, v in doc_files.items()}
                qrc_paths = {k : join(exp_folder, v) for k, v in qrc_files.items()}
                prompt_util.read_prompts(join("eloq/prompts"))

                print(f"\nCheck the responses of {llm_r} for {topic}\n")

                check_if_response_defused_confusion(llm_eval, doc_csv_schema, doc_paths["out"],
                                                qrc_csv_schema, qrc_paths[1], qrc_paths["out"], n=9, shot=5)
                end_time = utils.get_time()
                print(f"{topic} {llm_r} takes: {end_time - llm_start_time:.2f} seconds")
            end_time = utils.get_time()
            print(f"{topic} takes: {end_time - tpoic_start_time:.2f} seconds")