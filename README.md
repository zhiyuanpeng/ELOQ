<div align="center">

# ELOQ

[![Paper](https://img.shields.io/badge/paper-arxiv.2410.14567-B31B1B.svg
)](https://arxiv.org/pdf/2410.14567)
[![Conference](https://img.shields.io/badge/SIGIR-2025-B31B1B.svg?labelColor=%23FFEB3B&color=%231976D2
)](https://sigir2025.dei.unipd.it/)
[![Dataset](https://img.shields.io/badge/Huggingface-Datasets-B31B1B.svg?labelColor=%23FFD21E&color=%23FFD21E
)](https://huggingface.co/datasets/zhiyuanpeng/ELOQ)


</div>

## Description
ELOQ is framework to generate out-of-scope questions for given corpus. You can go through some examples via [ELOQ-Data-Viewer](https://huggingface.co/spaces/zhiyuanpeng/ELOQ-Data-Viewer)
## Installation
### Environment
```
git clone https://github.com/zhiyuanpeng/ELOQ.git
cd ELOQ
conda env create -f environment.yml
# install newscatcher
cd ./vendor/newscatcher-0.2.0
python setup.py install
```

### Configure .env file
Update the `.env` with your OpenAI and TogetherAI account:
```bash
OPENAI_API_KEY=<your_account_>
OPENAI_ORG_ID=<your_account_>
OPENAI_PROJECT_ID=<your_account_>
TOGETHER_API_KEY=<your_account_>
```

## Data

### Download the ELOQ dataset

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
# Download dataset as data and put within ELOQ
cd ELOQ
git clone https://huggingface.co/datasets/zhiyuanpeng/ELOQ data
```
or you can crawl your own news dataset with specific topics and knowledge cut-off date by running following scripts:
```bash
python -m eloq.src.crawling.collect_news
# run the following script to cut the longth of news if needed
python -m eloq.src.crawling.cut_news
```
ELOQ dataset consists the following files:

- news.json
    ```json
     "news_1": {
        "website": "yahoo.com",
        "url": "https://www.yahoo.com/news/batteries-walkie-talkies-exploded-lebanon-104205874.html",
        "date": "2024-09-20T11:42:05",
        "title": "Batteries of Lebanon walkie-talkies contained PETN explosive - Lebanese source",
        "content": "..."
     }
    ```
- questions.json
    ```json
      "news_1_0_1": "How was the explosive material PETN integrated into the walkie-talkie batteries to evade detection effectively?"
    ```
- silver_responses.json
    ```json
    "news_1_0_1": {
        "llm_confusion_label": llm_confusion_label,
        "gpt-3.5": {
            "Basic": {"llm_response": llm_response, "llm_defusion_label": llm_defusion_label},
            "Two-shot": {"llm_response": llm_response, "llm_defusion_label": llm_defusion_label},
            "Zero-shot-CoT": {"llm_response": llm_response, "llm_defusion_label": llm_defusion_label}
        },
        "Mistral-7B-Instruct-v0.3": {...},
        "Meta-Llama-3.1-70B-Instruct-Turbo": {...},
        "Meta-Llama-3.1-8B-Instruct-Turbo": {...},
        "Llama-3.2-3B-Instruct-Turbo": {...},
        "Meta-Llama-3.3-70B-Instruct-Turbo": {...},
    }
    ```
- golden_responses.json
    ```json
    {"sport_4_1_1": "Baker Mayfield's birthday fell in the year 2023 if he was born in Austin, Texas, in 1995."}
    ```
- silver.csv
    - doc_id,q_id,llm_confusion_label
- golden.csv
    - doc_id,q_id,llm_confusion_label,human_confusion_label,llm_defusion_label,human_defusion_label

### Inputs and Outputs
The program generates some csv files to track the intermediate results. 
#### CSV Files

- docs_in.csv
    - doc_id, source, document
- docs_0.csv 
    - doc_id,source,document,LLM_q,doc_prompt
- docs_1.csv
    - doc_id,source,document,LLM_q,doc_prompt,reduce_doc
- docs_2.csv
    - doc_id,source,document,LLM_q,doc_prompt,reduce_doc,modify_doc
- docs_3.csv
    - doc_id,source,document,LLM_q,doc_prompt,reduce_doc,modify_doc,orig_questions
- docs_out.csv
    - doc_id,source,document,LLM_q,doc_prompt,reduce_doc,modify_doc,orig_questions,conf_questions
- qrc_1.csv
    - doc_id,q_id,is_confusing,question,LLM_r,response
- qrc_2.csv
    - doc_id,q_id,is_confusing,question,LLM_r,response,confusion
- qrc_out.csv
    - doc_id,q_id,is_confusing,question,LLM_r,response,defusion,is_defused

#### Column Names

- doc_id: news id with format `{topic}_{number}.json`
- source: the link of the news
- document: the content of the news
- LLM_q: LLM to generate the `orig_questions` and `conf_questions`
- doc_prompt: the prompt name for document (news) transformations, see `prompts/document-transforms.json` for its content.
- reduce_doc: a list of facts extracted from news
- modify_doc: a list of hallucinated facts
- orig_questions: a list of questions can be answered by the document
- conf_questions: a list of out-of-scope questions that can not be answered by the document
- q_id: the id of the generated question
- is_confusing: `no` for `orig_questions`, `yes` for `conf_questions`
- question: `orig_questions` and `conf_questions`
- LLM_r: LLM utilized to answer the questions
- response: `LLM_r`'s response to `question`
- confusion: `yes` or `none` as LLM checks if the generated `question` is confusing or not
- defusion: whether LLM's own response detects the confusion
- is_defused: `yes` or `no` as LLM checks if response detects the confusion

## Question Generation

```python
python -m eloq.src.llm_question 
```

## Response Collection

```python
python -m eloq.src.llm_response
```
## Defuse

```python
python -m eloq.src.llm_defuse
```