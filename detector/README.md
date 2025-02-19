# Guide for Running and Testing the Embedding Classifiers and SBERT NLI baseline
To set up, you need 2 folders on the same level, next to this `token_embedding_classifier` folder
1. A folder called "data", which contains `v1.1/silver_train`, `v1.1/silver_val`, `v1.1/golden.csv`, etc. 
2. If you want to download the LLM from HuggingFace everytime, you can ignore this step, and just remember to include `hf_token` in the args. Otherwise please read on: You need a folder called `llms` if you want to use the LLMs downloaded from HuggingFace that are stored on your machine. The `llms` folder contains downloaded llms like `llms/meta-llama/Llama-3.2-3B-Instruct`, `llms/mistralai/Mistral-7B-Instruct-v0.3`, `llms/openai-community/gpt2`, etc. In these folders there should be the model weights and tokenizer etc. 

## Example commands
For `--which_embedding` we implemented 3 choices. Choose 1 from: 
1. "UnusedToken" 
2. "EOSToken"
3. "SentenceBERT"
### Training:
Load from HF and log as a debug experiment
```
python -m token_embedding_classifier.train
    --which_embedding "UnusedToken"
    --pretrained_model_name "meta-llama/Llama-3.2-3B-Instruct"
    --lr 1e-4 --batch_size 8 --epochs 10
    --hf_token "YOUR HF TOKEN"
    --exp_note "Running for the first time"
    --debug
```
Load from local and run experiment normally
```
python -m token_embedding_classifier.train
    --which_embedding "UnusedToken"
    --pretrained_model_name "meta-llama/Llama-3.2-3B-Instruct"
    --lr 1e-4 --batch_size 8 --epochs 10
    --exp_note "Running for the first time" 
```

### Testing:
Generic testing
```
python -m token_embedding_classifier.test
    --which_embedding "UnusedToken"
    --pretrained_model_name "meta-llama/Llama-3.2-3B-Instruct"
    --eval_file_name "golden.csv" 
```
For more options please look at token_embedding_classifier/test.py

