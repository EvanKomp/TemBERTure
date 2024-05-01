"""Temberture model

THis module was updated 5.1 by Evan Komp in order to make it a runnable script.

The script portion uses replica 1 of temberture TM and compute attentions, taking a fasta
input and writing the output as a csv file
"""
import os
import argparse
from transformers import BertTokenizer
from adapters import BertAdapterModel
#import logging
import tqdm 
import math
import numpy as np
import torch.nn as nn
import torch
#logger = logging.getLogger(__name__)

class TemBERTure:
    """
    This class initializes and utilizes a pretrained BERT-based model (model_name) with adapter layers tuned
    for classification or regression tasks. The adapter path (adapter_path) provides the pre-trained
    adapter and head for the specified model and task (regression or classification).

    Attributes:
        adapter_path (str): Path to pre-trained adapters and heads for the model.
        model_name (str, default='Rostlab/prot_bert_bfd'): Name of the BERT-based model.
        batch_size (int, default=16): Batch size for predictions.
        device (str, default='cuda'): Device for running the model ('cuda' or 'cpu').

    Methods:
        __init__: Initializes the TemBERTure class with the specified BERT-based model,
                adapter path, tokenizer, batch size, and device.
        predict: Takes input texts, tokenizes them, and predicts outputs (classification/regression)
                using the loaded model and its adapters.
    """
    def __init__(self, adapter_path, model_name='Rostlab/prot_bert_bfd',batch_size=16, device='cuda', task = 'regression'):
        self.model = BertAdapterModel.from_pretrained(model_name) 
        self.model.load_adapter(os.path.join(adapter_path,'AdapterBERT_adapter'), with_head=True)
        self.model.load_head(os.path.join(adapter_path, 'AdapterBERT_head_adapter'))
        self.model.set_active_adapters(['AdapterBERT_adapter'])
        self.model.active_head == 'AdapterBERT_head_adapter'  # pretrained for cls task adapter
        self.model.train_adapter(["AdapterBERT_adapter"])
        self.model.delete_head('default')
        self.model.bert.prompt_tuning = nn.Identity()
        #logger.info(f' * USING PRE-TRAINED ADAPTERS FROM: {adapter_path}')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.device = device
        self.task = task
    
    def predict(self, input_texts):
        self.model = self.model.to(self.device)
        input_texts = [" ".join("".join(sample.split())) for sample in input_texts]
        #input_texts = input_texts.tolist()
        nb_batches = math.ceil(len(input_texts) / self.batch_size)
        y_preds = []

        for i in tqdm.tqdm(range(nb_batches)):
            batch_input = input_texts[i * self.batch_size: (i+1) * self.batch_size]
            encoded = self.tokenizer(batch_input, truncation=True, padding=True, max_length=512, return_tensors="pt").to(self.device)
            y_preds += self.model(**encoded).logits.reshape(-1).tolist()

        if self.task == 'classification':
            preds = 1 / (1 + np.exp(-np.array(y_preds)))
            y_preds = (preds > 0.5).astype(int)# Trasforma le probabilità in etichette binarie
        
        status = 'Thermophilic' if y_preds[0] == 1 else 'Non-thermophilic'
        print('Predicted thermal class:', status)
        print('Thermophilicity prediction score:', preds[0])

        return [status,preds[0]]
    
    def compute_attention(self, input_texts):
        if self.batch_size != 1:
            raise ValueError("Batch size must be 1 for computing attentions. I don't want to keep track of the attention mask. sorry.")

        self.model = self.model.to(self.device)
        input_texts = [" ".join("".join(sample.split())) for sample in input_texts]
        #input_texts = input_texts.tolist()
        nb_batches = math.ceil(len(input_texts) / self.batch_size)
        attention_list = []

        for i in tqdm.tqdm(range(nb_batches)):
            batch_input = input_texts[i * self.batch_size: (i+1) * self.batch_size]
            encoded = self.tokenizer(batch_input, truncation=True, padding=True, max_length=512, return_tensors="pt").to(self.device)
            out = self.model(**encoded, output_attentions=True)
            
            # mean over attentions last layer
            # note that the first and last token are [CLS] and [SEP] respectively
            # we want to remove those
            attentions = out.attentions[-1].cpu().detach().numpy() # B x H x (L+2) x (L+2)
            attentions = np.mean(attentions, axis=1) # B x (L+2) x (L+2)
            attentions = attentions[0] # (L+2) x (L+2)
            attentions = np.mean(attentions, axis=0) # L+2
            attentions = attentions[1:-1] # L
            attention_list.append(attentions)

        return attention_list
    

def main(input_fasta: str, output_dir: str):
    """Main function to predict thermal class and compute attentions.
    
    In the output dir, there will be a csv file for each ID in the input fasta eg. <id>.csv
    """
    # read the fasta file
    with open(input_fasta, 'r') as f:
        lines = f.readlines()
    ids = []
    sequences = []
    for line in lines:
        if line.startswith('>'):
            ids.append(line[1:].strip())
        else:
            sequences.append(line.strip())

    # load the model. The weight can be found in the same folder as this script
    adapter_path = os.path.join(os.path.dirname(__file__), 'temBERTure_TM', 'replica1')
    temBERTure = TemBERTure(
        adapter_path=adapter_path,
        device="mps" if torch.backends.mps.is_available() else "cuda",
        task='regression',
        batch_size=1
    )

    attentions = temBERTure.compute_attention(sequences)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for id, attention in zip(ids, attentions):
        with open(os.path.join(output_dir, f'{id}.csv'), 'w') as f:
            f.write('position,attention\n')
            for i, att in enumerate(attention):
                f.write(f'{i},{att}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='temberTURE attentions')
    parser.add_argument('input_fasta', type=str, help='Input fasta file')
    parser.add_argument('output_dir', type=str, help='Output directory')
    args = parser.parse_args()
    main(args.input_fasta, args.output_dir)
    

    
    
    
