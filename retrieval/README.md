# Instructions to download the data and run retrieval experiments

2/18/2025: <br>
Please first create a folder named `data` directly under ELOQ. Then `cd` into the `data` folder and do 
```
git clone https://huggingface.co/datasets/zhiyuanpeng/ELOQ
```
to clone the dataset from huggingface. This is temporary as we are still working on making it loadable through load_dataset(). <br>

Then to run some retrieval results: 

```
python run.py --data_dir YOUR_WORKING_DIR/data --corpus 200 --model bm25
```

The argument --corpus can take "20", "200", and "full". 20 means 20 documents per topic, 200 means 200 documents per topic, and "full" means all documents for each topic. <br>
As of now, --model can take one value from ['stella', 'linq', 'bm25', 'sbert', 'bge']. 

- stella: NovaSearch/stella_en_1.5B_v5
- linq: Linq-AI-Research/Linq-Embed-Mistral
- bm25: LuceneBM25Model
- sbert: sentence-transformers/all-mpnet-base-v2
- bge: BAAI/bge-large-en-v1.5
  
The result will be printed in 2 parts: "In-scope" and "Out-of-scope", where each represent the retreival performance on each set of questions. Please look into `run.py` for how documents, queries, and qrels are loaded. 
