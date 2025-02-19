import os
from os.path import join
from detector.dataset import EOSTokenEmbeddingDataset, UnusedTokenEmbeddingDataset, SentenceBERTNLIFeatureDataset
from detector.model import EmbeddingClassifier
from detector.utils import seed_everything, get_current_commit_id
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix, BinaryAccuracy
from argparse import ArgumentParser


def start_MLflow_tracking(args):
    '''
    Set up for MLflow tracking
    Run "mlflow ui" in your command line to see MLflow UI 
    '''
    if args.debug:
        mlflow.set_experiment("debug")
    else:
        mlflow.set_experiment(f"{args.which_embedding}-{args.pretrained_model_name}")
    # read the git commit id as run_name
    commit_id = get_current_commit_id()
    mlflow.start_run(run_name=f"{commit_id[:7]}-{args.exp_note}")
    args_dict = vars(args)
    for key, value in args_dict.items():
        mlflow.log_param(key, value)

def load_model_and_tokenizer_from_local(model_path):
    '''
    Load LLM and tokenizer from  local path
    Models are loaded for infer only, with torch.bfloat16 dtype and output_hidden_states=True
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, output_hidden_states=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.cuda()
    model.eval()
    return tokenizer, model

def load_model_and_tokenizer_from_hf(model_name, hf_token): 
    '''
    Load LLM and tokenizer from HuggingFace. Must have huggingface token to access private models
    Models are loaded for infer only, with torch.bfloat16 dtype and output_hidden_states=True
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, output_hidden_states=True, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    model.cuda()
    model.eval()
    return tokenizer, model

def create_datasets(data_dir, tokenizer, llm, llm_name, dataset_class):
    '''
    Create train and eval datasets
    '''
    train_dataset = dataset_class(queries_path=join(data_dir, "v1.1", "questions.json"), 
        corpus_path=join(data_dir, "v1.1", "news.json"), 
        tsv_path=join(data_dir, "v1.1", "silver_train.csv"), 
        tokenizer=tokenizer, 
        llm=llm,
        llm_name=llm_name,
        prompt="Read the document and answer the question based on the document. \nDocument: ", 
        connection="\nQuestion: "
    )
    eval_dataset = dataset_class(queries_path=join(data_dir, "v1.1", "questions.json"), 
        corpus_path=join(data_dir, "v1.1", "news.json"), 
        tsv_path=join(data_dir, "v1.1", "silver_val.csv"), 
        tokenizer=tokenizer, 
        llm=llm,
        llm_name=llm_name,
        prompt="Read the document and answer the question based on the document. \nDocument: ", 
        connection="\nQuestion: "
    )
    return train_dataset, eval_dataset

def set_up(args, cwd): 
    '''
    Create train, eval datasets and classifier based on the args
    '''
    if args.which_embedding != "SentenceBERT":
        # Load LLM and tokenizer
        if args.hf_token:
            tokenizer, llm = load_model_and_tokenizer_from_hf(args.pretrained_model_name, args.hf_token)
        else:
            local_model_path = join(cwd, "llms", args.pretrained_model_name) # This dir should have the model and tokenizer files
            tokenizer, llm = load_model_and_tokenizer_from_local(local_model_path)
        # Load dataset
        data_dir = join(cwd, "data") # This dir should have the v1.1 folder with questions.json, news.json, silver_train.csv, and silver_val.csv
        if args.which_embedding == "UnusedToken":
            dataset_class = UnusedTokenEmbeddingDataset
        elif args.which_embedding == "EOSToken":
            dataset_class = EOSTokenEmbeddingDataset
        train_dataset, eval_dataset = create_datasets(data_dir, tokenizer, llm, args.pretrained_model_name, dataset_class)
        classifier = EmbeddingClassifier(embedding_size=llm.config.hidden_size).cuda()
        classifier.to(torch.bfloat16)
    else: # SentenceBERT
        train_dataset = SentenceBERTNLIFeatureDataset(queries_path=join(data_dir, "v1.1", "questions.json"), 
            corpus_path=join(data_dir, "v1.1", "news.json"), 
            tsv_path=join(data_dir, "v1.1", "silver_train.csv"), 
            model_name=args.pretrained_model_name
        )
        eval_dataset = SentenceBERTNLIFeatureDataset(queries_path=join(data_dir, "v1.1", "questions.json"), 
            corpus_path=join(data_dir, "v1.1", "news.json"), 
            tsv_path=join(data_dir, "v1.1", "silver_val.csv"), 
            model_name=args.pretrained_model_name
        )
        classifier = EmbeddingClassifier(embedding_size=train_dataset.model.config.hidden_size * 3).cuda() # *3 because [u; v; |u-v|]
    
    return train_dataset, eval_dataset, classifier

def train(train_dataloader, eval_dataloader, classifier, args, cwd): 
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_f1 = 0
    patient = 0
    best_classifier_save_path = ""
    # Training loop
    for epoch in range(args.epochs):
        classifier.train()
        step = 0
        for batch_of_embeddings, (_, _, labels)  in tqdm(train_dataloader):
            # Forward pass
            logits = classifier(batch_of_embeddings)
            # Compute loss
            loss = criterion(logits, torch.tensor(labels).unsqueeze(1).cuda())
            step += 1
            mlflow.log_metric("step_loss", loss.item(), step=step)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation loop
        classifier.eval()
        precision_metric = BinaryPrecision()
        recall_metric = BinaryRecall()
        f1_metric = BinaryF1Score()
        accuracy_metric = BinaryAccuracy()
        confusion_matrix = BinaryConfusionMatrix()
        step = 0
        all_queries, all_pos_documents, all_neg_documents = [], [], []
        all_query_ids, all_pos_document_ids, all_neg_document_ids = [], [], []
        all_preds = []
        with torch.no_grad():
            for batch_of_embeddings, (_, _, labels)  in tqdm(eval_dataloader):
                step += 1
                # Forward pass
                logits = classifier(batch_of_embeddings).squeeze().cpu()
                predictions = (logits >= 0.5).long()
                labels = torch.tensor(labels).long()
                precision_metric.update(logits, labels)
                accuracy_metric.update(logits, labels)
                recall_metric.update(logits, labels)
                f1_metric.update(logits, labels)
                confusion_matrix.update(logits, labels)
                all_preds.extend(predictions)

        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        f1 = f1_metric.compute().item()
        confusion_matrix = confusion_matrix.compute().numpy()
        accuracy = accuracy_metric.compute().item()
        print(f"Precision: {precision: .4f}, Recall: {recall: .4f}, F1: {f1: .4f}, Accuracy: {accuracy: .4f}")
        print(f"TN is {confusion_matrix[0][0]}, FP is {confusion_matrix[0][1]}, FN is {confusion_matrix[1][0]}, TP is {confusion_matrix[1][1]}")
        mlflow.log_metric("precision", precision, step=epoch)
        mlflow.log_metric("recall", recall, step=epoch)
        mlflow.log_metric("f1", f1, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)
        mlflow.log_metric("TN", confusion_matrix[0][0], step=epoch)
        mlflow.log_metric("FP", confusion_matrix[0][1], step=epoch)
        mlflow.log_metric("FN", confusion_matrix[1][0], step=epoch)
        mlflow.log_metric("TP", confusion_matrix[1][1], step=epoch)
        
        # Save classifier's state_dict if find better classifier
        save_path = join(cwd, "data", "experiments", "classifiers", args.which_embedding, args.pretrained_model_name.replace("/", "-"))
        os.makedirs(save_path, exist_ok=True)
        if f1 > best_f1:
            patient = 0
            best_f1 = f1
            if os.path.exists(best_classifier_save_path):
                os.remove(best_classifier_save_path)
            best_classifier_save_path = join(save_path, f"epoch_{epoch}.pt")
            torch.save(classifier.state_dict(), best_classifier_save_path)
        patient += 1
        if patient > 5:
            break
    
if __name__ == "__main__":
    seed_everything(42)
    
    parser = ArgumentParser()
    parser.add_argument("--which_embedding", type=str, default="UnusedToken", help="Which embedding to use, either 'EOS', 'UnusedToken', or 'SentenceBERT'")
    parser.add_argument("--pretrained_model_name", default="meta-llama/Llama-3.2-3B-Instruct", type=str, help="Instruct-tuned LLMs or SentenceBERT models")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate")
    parser.add_argument("--weight_decay", default=1e-9, type=float, help="Weight decay rate")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--hf_token", default="", type=str, help="If you provide a HuggingFace token, the LLM will be loaded from HuggingFace")
    parser.add_argument("--exp_note", default="default experiment note", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cwd = os.getcwd()
    
    start_MLflow_tracking(args)
    
    # Dataset and Classifier Setup
    train_dataset, eval_dataset, classifier = set_up(args, cwd)
    
    # DataLoaders Setuo
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)
    
    # Train
    train(train_dataloader, eval_dataloader, classifier, args, cwd)
    
    mlflow.end_run()