import os
import math
from sentence_transformers.readers.InputExample import InputExample
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import random
from torch.utils.data import DataLoader
from dataset import CrossEncoderDataset

use_cuda = torch.cuda.is_available()
max_seq_length = 256
batch_size = 12
num_epochs = 30
model_name = 'cross-encoder/stsb-distilroberta-base'


os.makedirs('./output', exist_ok=True)
cross_encoder_path = f'./output/cross_{model_name}/'
bi_encoder_path = f'./output/bi_{model_name}/'
    
if __name__ == '__main__':
    _dataset = CrossEncoderDataset(iter=50000)
    test_dataset = _dataset.test()

    evaluator = CECorrelationEvaluator.from_input_examples(test_dataset, name='sts-dev')

    cross_train_data_loader = DataLoader(_dataset, batch_size=batch_size, num_workers=0)
    cross_encoder = CrossEncoder(model_name)
    warmup_steps = math.ceil(len(cross_train_data_loader) * num_epochs * 0.1) 
    cross_encoder.fit(train_dataloader=cross_train_data_loader, evaluator=evaluator, evaluation_steps=1000, epochs=num_epochs, warmup_steps=warmup_steps, output_path=cross_encoder_path, use_amp=True)
    cross_encoder.save(cross_encoder_path)
    crossencoder = CrossEncoder(cross_encoder_path)

    emb_csv = pd.read_csv(os.path.join('baseline1', 'test.csv'), usecols=['label', 'cap_idx'], sep='\t').reset_index()
    test_data = np.load('./baseline1/caption_text.npy', allow_pickle=True)

    positive_samples = emb_csv[(emb_csv['label'] == 1)]
    negative_samples = emb_csv[emb_csv['label'] == -1]
    neutral_samples = emb_csv[emb_csv['label'] == 0 | (emb_csv['label'] == 1)]

    positive_sentence_pair = []

    def process_samples(df):
        sentence_pair = []
        for i in tqdm(range(1, df['cap_idx'].shape[0]-2, 1), desc='processing samples'):
            idx_1 = df.iloc[i]['cap_idx']
            idx_2 = df.iloc[i-1]['cap_idx']
            try:
                data_1 = test_data[idx_1][0]
                data_2 = test_data[idx_2][0]
                sentence_pair.append([data_1, data_2])
            except:
                continue
        return sentence_pair



    positive_sentence_pair = process_samples(positive_samples)
    positive_scores = crossencoder.predict(positive_sentence_pair, show_progress_bar=True)
    positive_class = [1 if score >= 0.5 else 0 for score in positive_scores]

    negative_sentence_pair = process_samples(negative_samples)
    negative_scores = crossencoder.predict(negative_sentence_pair, show_progress_bar=True)
    negative_class = [1 if score >= 0.5 else 0 for score in negative_scores]

    neutral_sentence_pair = process_samples(neutral_samples)
    neutral_scores = crossencoder.predict(neutral_sentence_pair, show_progress_bar=True)
    neutral_class = [1 if score >= 0.5 else 0 for score in neutral_scores]

    TP = np.count_nonzero(positive_class)
    TN = np.count_nonzero(negative_class)

    FP = len(positive_class) - TP
    FN = len(negative_class) - TN

    recall = TP / (TP + FN)
    precs = TP / (TP + FP)
    F1 = 2 * (precs * recall) / (precs + recall)
    print("RECALL: ", recall, "PRECISION: ", precs, "F1: ", F1)
    print(f"CONFUSION MATRIX: \t {TP}, {FP} \n\t\t {FN}, {TN}" )


    print("\nNEUTRAL")

    TP = np.count_nonzero(neutral_class)
    TN = np.count_nonzero(negative_class)

    FP = len(neutral_class) - TP
    FN = len(negative_class) - TN

    recall = TP / (TP + FN)
    precs = TP / (TP + FP)
    F1 = 2 * (precs * recall) / (precs + recall)
    print("RECALL: ", recall, "PRECISION: ", precs, "F1: ", F1)
    print(f"CONFUSION MATRIX: \t {TP}, {FP} \n\t\t {FN}, {TN}" )
