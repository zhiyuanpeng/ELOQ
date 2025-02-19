import os
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from os.path import join
import argparse
# Ensure the necessary NLTK data files are downloaded
nltk.download('punkt')
cwd = os.getcwd()

def process_json_file(file_path, output_dir, cutt_length):
    with open(file_path, 'r') as f:
        data = json.load(f)
    content = data.get('content', '')
    token_count = data.get('token_count', 0)
    if token_count < cutt_length:
        new_content_str = content
        token_count = token_count
    else:
        sentences = sent_tokenize(content)
        new_content = []
        token_count = 0
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            token_count += len(tokens)
            new_content.append(sentence)
            if token_count > cutt_length:
                break
        new_content_str = ' '.join(new_content)
    data['content'] = new_content_str
    data['token_count'] = token_count
    del data['summary']
    output_file_path = os.path.join(output_dir, os.path.basename(file_path))
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_folders(base_dir, output_name):
    parent_dir = os.path.dirname(base_dir)
    for root, dirs, files in os.walk(base_dir):
        topic = os.path.basename(root)
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                output_dir = os.path.join(parent_dir, output_name, topic)
                os.makedirs(output_dir, exist_ok=True)
                process_json_file(file_path, output_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cut news articles to limit tokens')
    parser.add_argument('dataset_name', help='dataset name of the raw new articles', default='News1k2024', type=str)
    parser.add_argument('cutt_length', help='cutt length of the news articles', default=300, type=int)
    args = parser.parse_args()

    base_dir = join(cwd, 'data', 'raw', args.dataset_name)
    output_dir = join(cwd, 'data', 'processed', f'{args.dataset_name}_{args.cutt_length}')
    process_folders(base_dir, f'{args.dataset_name}_{args.cutt_length}')