import pandas as pd
import os
from tqdm import tqdm
from os.path import join, dirname, basename
from dotenv import load_dotenv
load_dotenv()
import eloq.src.utils.prompt_util as prompt_util
import eloq.src.utils.doc_util as utils
from eloq.src.utils.doc_util import doc_csv_schema, qrc_csv_schema, doc_files, qrc_files

def record_llm_and_prompts(llm, doc_prompt, schema, path_in, path_out):
    """
    Record LLM(s) and prompt(s) into the CSV table to use for generating out-of-scope questions
    """
    df = utils.read_csv(path_in, "Read the input document table from CSV file")
    # Check that the correct columns are present in the CSV
    assert schema["document"] in set(df.columns)
    assert schema["LLM_q"] not in set(df.columns)
    assert schema["doc_prompt"] not in set(df.columns)
    df = df.reindex(columns = df.columns.tolist() + [schema["LLM_q"], schema["doc_prompt"]])
    df = df.astype({schema["LLM_q"]: str, schema["doc_prompt"]: str}, copy = False)
    for row_id, row in tqdm(df.copy().iterrows(), total = df.shape[0]):
        df.loc[row_id, schema["LLM_q"]] = llm
        df.loc[row_id, schema["doc_prompt"]] = doc_prompt
    utils.write_csv(df, path_out, "Write the document table with LLM names and prompt keys to CSV file")

def reduce_original_documents(schema, num_fact, path_in, path_out):
    """
    Use LLM (or other means) to create a reduced version for each document
    """
    df = utils.read_csv(path_in, "Read the input document table from CSV file")
    # Check that the correct columns are present in the CSV
    assert schema["document"] in set(df.columns)
    assert schema["LLM_q"] in set(df.columns)
    assert schema["doc_prompt"] in set(df.columns)
    assert schema["reduce_doc"] not in set(df.columns)

    df = df.reindex(columns = df.columns.tolist() + [schema["reduce_doc"]])
    df = df.astype({schema["reduce_doc"]: str}, copy = False)
    print(f"Use LLM to create a reduced version for each document from column {schema['document']}")
    for row_id, row in tqdm(df.copy().iterrows(), total = df.shape[0]):
        llm = row[schema["LLM_q"]]
        prompt_key = row[schema["doc_prompt"]]
        document = utils.prepare_document(row[schema["document"]])
        reduce_doc = prompt_util.reduce_document(llm, document, num_fact, prompt_key)
        df.loc[row_id, schema["reduce_doc"]] = reduce_doc
        # break
    utils.write_csv(df, path_out, "Write the document table with the reduced versions to CSV file")


def modify_reduced_documents(schema, num_fact, path_in, path_out):
    """
    Ask LLM to guess the missing facts
    """
    df = utils.read_csv(path_in, "Read the reduced document table from CSV file")
    # Check that the correct columns are present in the CSV
    assert schema["LLM_q"] in set(df.columns)
    assert schema["doc_prompt"] in set(df.columns)
    assert schema["reduce_doc"] in set(df.columns)
    assert schema["modify_doc"] not in set(df.columns)

    df = df.reindex(columns = df.columns.tolist() + [schema["modify_doc"]])
    df = df.astype({schema["modify_doc"]: str}, copy = False)
    print(f"Use LLM to guess the missing facts from column {schema['reduce_doc']}")
    for row_id, row in tqdm(df.copy().iterrows(), total = df.shape[0]):
        llm = row[schema["LLM_q"]]
        prompt_key = row[schema["doc_prompt"]]
        reduce_doc = utils.prepare_document(row[schema["reduce_doc"]])
        document = utils.prepare_document(row[schema["document"]])
        modify_doc = prompt_util.modify_reduced_document(llm, document, reduce_doc, num_fact, prompt_key)
        df.loc[row_id, schema["modify_doc"]] = modify_doc
        # break
    utils.write_csv(df, path_out, "Write the document table with the modified versions of reduced docs to CSV file")

def generate_questions_for_documents(num_q, schema, col_refs, path_in, path_out):
    """
    For each original document, ask LLM to write `num_q` questions answered in the document
    """
    doc_ref = col_refs[0]
    que_ref = col_refs[1]
    df = utils.read_csv(path_in, "Read the document table from CSV file")
    # Check that the correct columns are present in the CSV
    assert schema["LLM_q"] in set(df.columns)
    assert schema[doc_ref] in set(df.columns)
    assert schema[que_ref] not in set(df.columns)

    df = df.reindex(columns = df.columns.tolist() + [schema[que_ref]])
    df = df.astype({schema[que_ref]: str}, copy = False)
    print(f"Generate {num_q} questions for each document from column {schema[doc_ref]}")
    for row_id, row in tqdm(df.copy().iterrows(), total = df.shape[0]):
        llm = row[schema["LLM_q"]]
        document = utils.prepare_document(row[schema[doc_ref]])
        questions = prompt_util.generate_questions(llm, document, num_q)
        df.loc[row_id, schema[que_ref]] = "\n".join([f"{i}. {q}" for i, q in enumerate(questions, start = 1)])
        # break
    utils.write_csv(df, path_out, "Write the document table with questions to CSV file")

def generate_confused_questions_for_documents(num_q, schema, col_refs, path_in, path_out):
    """
    Convert each hellucianted fact into a question which can only be answered by the hallucinated fact itself and can't be answered by the original document
    """
    doc_ref = col_refs[0]
    que_ref = col_refs[1]
    df = utils.read_csv(path_in, "Read the document table from CSV file")
    # Check that the correct columns are present in the CSV
    assert schema["LLM_q"] in set(df.columns)
    assert schema[doc_ref] in set(df.columns)
    assert schema[que_ref] not in set(df.columns)

    df = df.reindex(columns = df.columns.tolist() + [schema[que_ref]])
    df = df.astype({schema[que_ref]: str}, copy = False)
    print(f"Generate {num_q} questions for each document from column {schema[doc_ref]}")
    for row_id, row in tqdm(df.copy().iterrows(), total = df.shape[0]):
        llm = row[schema["LLM_q"]]
        document = utils.prepare_document(row[schema[doc_ref]])
        hallucinated_facts = utils.prepare_document(row[schema["modify_doc"]])
        questions = prompt_util.confuse_questions_v2(llm, document, hallucinated_facts= hallucinated_facts)
        df.loc[row_id, schema[que_ref]] = "\n".join([f"{i}. {q}" for i, q in enumerate(questions, start = 1)])
        # break
    utils.write_csv(df, path_out, "Write the document table with questions to CSV file")

def generate_RAG_responses(llm, doc_schema, doc_path, qr_schema, qr_path):
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
            response_q = prompt_util.generate_response(llm, document, q)
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
    backup_path = join(dirname(qr_path), f"llmr-{llm}", basename(qr_path))
    os.makedirs(dirname(backup_path), exist_ok = True)
    utils.write_csv(df_out, backup_path, "Write the question-response table to CSV file")

def create_dictionary_of_indexed_documents(doc_schema, doc_path):
    df_doc = utils.read_csv(doc_path, "Read the document table from CSV file")
    print("Create a dictionary of indexed documents")
    documents = {
        row[doc_schema["doc_id"]] : row[doc_schema["document"]]
            for _, row in tqdm(df_doc.iterrows(), total = df_doc.shape[0])
    }
    return documents

def find_confusion_in_question(llm, doc_schema, doc_path, qr_schema, qr_path_in, qr_path_out, n=1):
    documents = create_dictionary_of_indexed_documents(doc_schema, doc_path)
    df_qr = utils.read_csv(qr_path_in, "Read the question-response table from CSV file")    
    print("Ask LLM to find a false assumption in each question, or say 'none'")
    rows_out = []
    for _, row in tqdm(df_qr.iterrows(), total = df_qr.shape[0]):
        doc_id = row[qr_schema["doc_id"]]
        document = documents[doc_id]
        question = row[qr_schema["question"]]
        # llm = row[qr_schema["LLM_r"]]
        confusion = prompt_util.find_confusion(llm, document, question, n)
        row_out = dict(row)
        row_out[qr_schema["confusion"]] = confusion
        rows_out.append(row_out)
    df_out = pd.DataFrame.from_dict(rows_out, dtype = str)
    utils.write_csv(df_out, qr_path_out, "Write the question-response table to CSV file")

if __name__ == "__main__":
    start_time = utils.get_time()
    # Note: LLM_q may be stronger than LLM_r, to create more challenging confusions.
    llm_q = "gpt-4o-mini"  # LLM for generating questions (stronger)
    llm_r = "gpt-3.5"  # LLM for generating responses (weaker)
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
    for topic in topics:
        data_folder = f"data/News1k2024-300/{topic}/{news_num}"
        exp_folder = f"data/experiments/llmq-{llm_q}/docp-{doc_prompt}/{news_num}/{topic}"
        os.makedirs(exp_folder, exist_ok = True)
        doc_paths = {k : join(data_folder, v) if k == "in" else join(exp_folder, v)  for k, v in doc_files.items()}
        qrc_paths = {k : join(exp_folder, v) for k, v in qrc_files.items()}
        prompt_util.read_prompts(join("eloq/prompts"))

        print(f"\nSTEP 0: Record LLM(s) and prompt(s) to use for generating out-of-scope questions\n")

        record_llm_and_prompts(llm_q, doc_prompt, doc_csv_schema, doc_paths["in"], doc_paths[0])
        
        print(f"\nSTEP 1: Use LLM (or other means) to create a reduced version for each document\n")
        # F.1 Extract Claims Prompt
        reduce_original_documents(doc_csv_schema, num_fact, doc_paths[0], doc_paths[1])

        print(f"\nSTEP 2: Ask LLM to guess the missing facts\n")
        # F.2 Recover Missing Claims Prompt
        modify_reduced_documents(doc_csv_schema, num_fact, doc_paths[1], doc_paths[2])

        print(f"\nSTEP 3: For each original document, ask LLM to write " +
            f"{num_q_orig} questions answered in the document\n")
        # F.9 In-scope Question Generation Prompt
        generate_questions_for_documents(num_q_orig, doc_csv_schema, ["document", "orig_qs"],
                                        doc_paths[2], doc_paths[3])

        print(f"\nSTEP 4: For each modified reduced document, ask LLM to write at most " +
            f"{num_fact} questions that can't be answered in the document\n")
        # F.10 Out-of-scope Question Generation Prompt
        generate_confused_questions_for_documents(num_fact, doc_csv_schema, ["document", "conf_qs"], doc_paths[3], doc_paths["out"])

        print("\nSTEP 5: Give LLM the document and the question and record LLM's response\n")
        # F.6 RAG Basic
        generate_RAG_responses(llm_r, doc_csv_schema, doc_paths["out"], qrc_csv_schema, qrc_paths[1])

        print("\nSTEP 6: Ask LLM to find the confusion in each question\n")
        # F.3 Out-of-scope Judgement Prompt
        find_confusion_in_question(llm_eval, doc_csv_schema, doc_paths["out"],
                                            qrc_csv_schema, qrc_paths[1], qrc_paths[2], n=9)
        end_time = utils.get_time()
        print(f"{topic} takes: {end_time - start_time:.2f} seconds")