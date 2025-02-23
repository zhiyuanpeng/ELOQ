import streamlit as st
import os
import json
from os.path import join
import pandas as pd
from huggingface_hub import snapshot_download
import collections

@st.cache_data
def download_folder():
    local_dir = "view"
    os.makedirs(local_dir, exist_ok=True)
    st.info("Downloading folder from Hugging Face...It may take a while.")
    snapshot_download(
        repo_id="zhiyuanpeng/ELOQ",
        repo_type="dataset",
        local_dir=local_dir,  # Or a folder of your choice
        use_auth_token=False,
        allow_patterns=["ELOQ/*"]
    )
    return local_dir

@st.cache_data
def load_data():
    data_dir = "view/ELOQ"
    news_file = join(data_dir, "news.json")
    with open(news_file, "r") as f:
        news_data = json.load(f)
    question_file = join(data_dir, "questions.json")
    with open(question_file, "r") as f:
        question_data = json.load(f)
    silver_file = join(data_dir, "silver.csv")
    silver_data = pd.read_csv(silver_file)
    gold_file = join(data_dir, "gold.csv")
    gold_data = pd.read_csv(gold_file)
    silver_response_file = join(data_dir, "silver_responses.json")
    with open(silver_response_file, "r") as f:
        silver_response_data = json.load(f)
    gold_response_file = join(data_dir, "gold_responses.json")
    with open(gold_response_file, "r") as f:
        gold_response_data = json.load(f)
    silver_data = process_data(silver_data, "silver")
    gold_data = process_data(gold_data, "gold")
    return news_data, question_data, silver_data, gold_data, silver_response_data, gold_response_data

def process_data(data, file_name):
    # {data_dict: {q_id: {}, ...}}
    data_dict = collections.defaultdict(dict)
    for index, row in data.iterrows():
        doc_id = row["doc_id"]
        q_id = row["q_id"]
        if file_name == "gold":
            data_dict[doc_id][q_id] = {
                "llm_confusion_label": row["llm_confusion_label"],
                "human_confusion_label": row["human_confusion_label"],
                "llm_defusion_label": row["llm_defusion_label"],
                "human_defusion_label": row["human_defusion_label"]
            }
        else:
            data_dict[doc_id][q_id]  = {
                "llm_confusion_label": row["llm_confusion_label"]
            }
    return data_dict

def sidebar_logic(gold_doc_ids, silver_doc_ids):
    st.sidebar.header("Which data file to view?")
    file_name = st.sidebar.selectbox("Choose Data Name:", ["gold", "silver"])
    if file_name == "gold":
        doc_ids = gold_doc_ids
    else:
        doc_ids = silver_doc_ids
    topic_q_ids = collections.defaultdict(list)
    for doc_id in doc_ids:
        topic_q_ids[doc_id.split("_")[0]].append(doc_id)
    tpoic = st.sidebar.selectbox("Choose Doc id:", topic_q_ids.keys())
    doc_id = st.sidebar.selectbox("Choose Doc id:", topic_q_ids[tpoic])
    return file_name, doc_id

def init():
    st.set_page_config(layout="wide")
    cwd = os.getcwd() 
    st.title("Dataset Viewer")

    if "document_content" not in st.session_state:
        st.session_state.document_content = ""
    if "question_content" not in st.session_state:
        st.session_state.question_content = ""
        
    st.markdown(
        """
        <style>
        /* Make the left column sticky */
        div[data-testid="column"]:nth-child(1) {
            position: sticky;
            top: 0;
            height: 100vh;
            overflow-y: auto;
            padding-right: 10px;
            padding-bottom: 30px;
            border-right: 2px solid #ccc;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    return cwd 

def show_doc_contents(doc_data, doc_id):
    st.session_state.document_content = doc_data[doc_data["doc_id"] == doc_id]["document"].values[0]
    if st.session_state.document_content:
        st.write(f"### Document: {doc_id}")
        st.write(st.session_state.document_content)

def show_question_contents_and_labels(q_labels, question_data, gold_response_data):
    for q_id, labels in q_labels.items():
        st.write(f"### Question: {q_id}")
        st.text_area("Question:", value=question_data[q_id], key=q_id)
        if q_id in gold_response_data:
            st.text_area("Gold Response:", value=gold_response_data[q_id], key=f"gold_response_{q_id}")
        # Gather label info
        label_lines = []
        for label_key in [
            "llm_confusion_label",
            "human_confusion_label",
            "llm_defusion_label",
            "human_defusion_label"
        ]:
            if label_key in labels:
                label_lines.append(f"{label_key}: {labels[label_key]}")

        # Display all labels in one box
        if label_lines:
            st.text_area(
                "Labels:",
                value="\n".join(label_lines),
                key=f"labels_{q_id}"
            )
    

if __name__ == "__main__":

    cwd = init()
    # Check if the folder already exists
    # If not, download it
    if not os.path.exists("view"):
        folder_path = download_folder()
        st.success(f"Downloaded to {folder_path}")
    # Load data
    news_data, question_data, silver_data, gold_data, silver_response_data, gold_response_data = load_data()
    # Sidebar
    file_name, doc_id = sidebar_logic(gold_data.keys(), silver_data.keys())
    # Show Question contents
    left, right = st.columns([2 , 1.5])  # these numbers represent proportions
    # Display document
    with left:
        st.session_state.document_content = news_data[doc_id]["title"] + "\n" + news_data[doc_id]["content"]
        st.write(f"### Document: {doc_id}")
        st.write(st.session_state.document_content)
    # Display question
    with right:
        if file_name == "gold":
            q_labels = gold_data[doc_id]
        else:
            q_labels = silver_data[doc_id]
        show_question_contents_and_labels(q_labels, question_data, gold_response_data)


    


