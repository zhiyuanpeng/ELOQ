import os
from os.path import join
from detector.dataset import EOSTokenEmbeddingDataset, UnusedTokenEmbeddingDataset, SentenceBERTNLIFeatureDataset
from detector.model import EmbeddingClassifier
from detector.utils import seed_everything, get_current_commit_id, print_model_size
from detector.train import load_model_and_tokenizer_from_local, load_model_and_tokenizer_from_hf
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix, BinaryAccuracy
from argparse import ArgumentParser



def set_up_for_test(args, cwd):
    '''
    Create test dataset and load classifier based on the args
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
        test_dataset = dataset_class(queries_path=join(data_dir, "v1.1", "questions.json"), 
            corpus_path=join(data_dir, "v1.1", "news.json"), 
            tsv_path=join(data_dir, "v1.1", args.eval_file_name),
            tokenizer=tokenizer, 
            llm=llm,
            llm_name=args.pretrained_model_name,
            prompt="Read the document and answer the question based on the document. \nDocument: ", 
            connection="\nQuestion: "
        )
        classifier = EmbeddingClassifier(embedding_size=llm.config.hidden_size).cuda()
    else: # SentenceBERT
        test_dataset = SentenceBERTNLIFeatureDataset(queries_path=join(data_dir, "v1.1", "questions.json"), 
            corpus_path=join(data_dir, "v1.1", "news.json"), 
            tsv_path=join(data_dir, "v1.1", args.eval_file_name),
            model_name=args.pretrained_model_name
        )
        classifier = EmbeddingClassifier(embedding_size=test_dataset.model.config.hidden_size * 3).cuda() # *3 because [u; v; |u-v|]
    
    # Dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, drop_last=False)
    
    # Load the classifier from the best save weights
    if args.best_classifier_save_path: 
        best_classifier_path = args.best_classifier_save_path
    else: # Load the best model from the experiments folder
        save_path = join(cwd, "data", "experiments", "classifiers", args.which_embedding, args.pretrained_model_name.replace("/", "-"))
        for file in os.listdir(save_path):
            if file.endswith(".pt"):
                best_classifier_path = join(save_path, file)
                break
    classifier.load_state_dict(torch.load(best_classifier_path))
    if args.which_embedding != "SentenceBERT":
        classifier.to(torch.bfloat16)
    print_model_size(classifier)
    
    return test_dataloader, classifier

def evaluate(test_dataloader, classifier, args, cwd):
    classifier.eval()
    precision_metric = BinaryPrecision()
    recall_metric = BinaryRecall()
    f1_metric = BinaryF1Score()
    accuracy_metric = BinaryAccuracy()
    confusion_matrix = BinaryConfusionMatrix()
    step = 0
    all_queries, all_pos_documents, all_neg_documents = [], [], []
    all_query_ids, all_pos_document_ids, all_neg_document_ids = [], [], []
    all_preds, wrong_predictions = [], []
    with torch.no_grad():
        for batch_of_eos_embedding, (q_ids, doc_ids, labels)  in tqdm(test_dataloader, desc="Evaluating"):
            step += 1
            # Forward pass
            logits = classifier(batch_of_eos_embedding).squeeze().cpu()
            predictions = (logits >= 0.5).long()
            labels = torch.tensor(labels).long()
            precision_metric.update(logits, labels)
            accuracy_metric.update(logits, labels)
            recall_metric.update(logits, labels)
            f1_metric.update(logits, labels)
            confusion_matrix.update(logits, labels)
            all_preds.extend(predictions)
        
            if args.show_wrong_predictions:
                for i, prediction in enumerate(predictions):
                    if prediction != labels[i]:
                        wrong_predictions.append((q_ids[i], doc_ids[i], labels[i].item(), prediction.item()))

    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()
    confusion_matrix = confusion_matrix.compute().numpy()
    accuracy = accuracy_metric.compute().item()
    print(f"Precision: {precision: .4f}, Recall: {recall: .4f}, F1: {f1: .4f}, Accuracy: {accuracy: .4f}")
    print(f"TN is {confusion_matrix[0][0]}, FP is {confusion_matrix[0][1]}, FN is {confusion_matrix[1][0]}, TP is {confusion_matrix[1][1]}")

    if args.show_wrong_predictions:
        for q_id, doc_id, label, pred in wrong_predictions:
            print(f"Query ID: {q_id}, Document ID: {doc_id}, Label: {label}, Prediction: {pred} (1: confusing, 0: not confusing)")
            query = dataset.queries[q_id]
            document = dataset.corpus[doc_id]['title'] + " " + dataset.corpus[doc_id]['content']
            print(f"Query: {query}\n\nDocument: {document}")
            print("====================================================")
        print(f"Total wrong predictions: {len(wrong_predictions)}")
    
    
if __name__ == "__main__":
    seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument("--which_embedding", type=str, default="UnusedToken", help="Which embedding to use, either 'EOS', 'UnusedToken', or 'SentenceBERT'")
    parser.add_argument("--pretrained_model_name", default="meta-llama/Llama-3.2-3B-Instruct", type=str, help="Instruct-tuned LLMs or SentenceBERT models")
    parser.add_argument("--eval_file_name", default="silver_test.csv", type=str)
    parser.add_argument("--hf_token", default="", type=str)
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--best_classifier_save_path", default="", type=str)
    parser.add_argument("--show_wrong_predictions", action='store_true')
    parser.add_argument("--exp_note", default="default experiment note", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cwd = os.getcwd()
    
    test_dataloader, classifier = set_up_for_test(args, cwd)

    evaluate(test_dataloader, classifier, args, cwd)
