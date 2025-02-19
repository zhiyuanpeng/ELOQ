# Instructions to download the data and run retrieval experiments

Please first create a folder named `data` directly under ELOQ. Then `cd` into the `data` folder and do `git clone https://huggingface.co/datasets/zhiyuanpeng/ELOQ` to clone the dataset from huggingface. This is temporary as we are still working on making it loadable through load_dataset(). <br>

Then to run some retrieval results: 

```
python run.py --data_dir YOUR_WORKIN_DIR\data --corpus 200 --model bm25
```

The result will be printed in 2 parts: "In-scope" and "Out-of-scope", where each represent the retreival performance on each set of questions. Please look into `run.py` for how documents, queries, and qrels are loaded. 
