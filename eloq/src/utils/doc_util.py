import pandas as pd
import re
import time

def read_csv(path, comment):
    print(comment + ":\n    " + path)
    df = pd.read_csv(path, dtype = str, na_filter = False)
    print("    Rows: " + str(len(df)) + ",  Cols: " + str(len(df.columns)))
    print("    " + str(df.columns))
    return df

def write_csv(df, path, comment):
    print(comment + ":\n    " + path)
    print("    Rows: " + str(len(df)) + ",  Cols: " + str(len(df.columns)))
    print("    " + str(df.columns))
    df.to_csv(path, index = False)
       
def prepare_document(raw_document):
    document = re.sub(r"\n\s*\n", "\n", raw_document)  # Remove excessive empty lines
    return document

def enum_list(questions, start=1):
    return "\n".join([f"{i}. {q}" for i, q in enumerate(questions, start = start)])

def parse_numbered_questions(text): # , min_number_of_items = 2):
    lines = text.splitlines()
    questions = []
    chunks = [] # One question could span multiple lines
    def add_question_from_chunks():
        nonlocal chunks, questions
        if chunks:
            question = (" ".join(chunks)).strip()
            if question[-1] == '?':
                questions.append(question)
            chunks = []
    for raw_line in lines:
        line = raw_line.strip()
        if len(line) > 0:
            x = re.search(r"^\d+[:\.]\s+", line)
            if x:  # The line starts a new question
                add_question_from_chunks()
                line = line[x.span()[1]:]
            chunks.append(line)
            if line[-1] == '?':
                add_question_from_chunks()
    add_question_from_chunks()
    # print(questions)
    return questions

def get_time():
    return time.time()

doc_csv_schema = {
        "doc_id" : "doc_id",           # Column with a unique document ID
        "source" : "source",           # Column with document source (e.g. URL)
        "document" : "document",       # Column with the text of the original document
        "LLM_q" : "LLM_q",             # Column with name of the LLM used to generate confusing questions
        "doc_prompt" : "doc_prompt",   # Column with JSON key of the prompt used to transform document
        "reduce_doc" : "reduce_doc",   # Column with the text of the reduced / simplified document
        "modify_doc" : "modify_doc",   # Column with the text of the modified / confused document
        "expand_doc" : "expand_doc",   # Column with the text of the expanded / reconstructed document
        "orig_qs" : "orig_questions",  # Column with the questions to original document (non-confusing)
        "conf_qs" : "conf_questions"   # Column with the questions to expanded document (confusing)
    }

qrc_csv_schema = {
    "doc_id" : "doc_id",           # Column with document ID, same as the other table
    "q_id" : "q_id",               # Column with question ID (for this document)
    "is_conf" : "is_confusing",    # Column with "yes" or "no" indicating if the question is confusing
    "question" : "question",       # Column with the question (either original or confusing)
    "LLM_r" : "LLM_r",             # Column with name of the LLM that generated the responses
    "response" : "response",       # Column with response generated given the document and the question
    "confusion" : "confusion",     # Column with LLM-found confusion in the question (or "none")
    "defusion" : "defusion",       # Column with LLM's reply on whether its own response detected the confusion
    "is_defused" : "is_defused"    # Column with "yes" or "no" as LLM checks if response detected the confusion
}

doc_files = {
            "in" : "docs_in.csv",
            "out" : "docs_out.csv",
            0 : "docs_0.csv",
            1 : "docs_1.csv",
            2 : "docs_2.csv",
            3 : "docs_3.csv",
            4 : "docs_4.csv"
        }
qrc_files = {
    "out" : "qrc_out.csv",
    "filter" : "qrc_filter.csv",
    1 : "qrc_1.csv",
    2 : "qrc_2.csv"
}


