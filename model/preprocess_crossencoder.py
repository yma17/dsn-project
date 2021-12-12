import numpy as np
from numpy.core.numeric import cross
import pandas as pd
from sentence_transformers import models, losses,  CrossEncoder
from tqdm import trange


cross_model_name = 'cross-encoder/stsb-distilroberta-base'
cross_encoder_path = f'./output/cross_{cross_model_name}/'
cross_encoder = CrossEncoder(cross_encoder_path)
baseline1 = pd.read_csv("./baseline1/_embedd.csv", usecols=['video_id', 'label', 'cap_idx'], sep='\t')
baseline1_txt = np.load('./baseline1/caption_text.npy', allow_pickle=True)
unsupervised_set = pd.read_csv(f'./scrubbed/_embedd.csv', usecols=['video_id', 'cap_idx'], sep='\t', low_memory=True) 
unsupervised_txt = np.load('./scrubbed/caption_text.npy', allow_pickle=True)

positive_samples = baseline1[(baseline1['label'] == 1)]
negative_samples = baseline1[baseline1['label'] == -1]
neutral_samples = baseline1[baseline1['label'] == 0]


set_size = unsupervised_set['cap_idx'].shape[0]
def calculate_score(df):
    _labels = [-2 for i in range(set_size)]
    for i in trange(set_size, desc='calculating cross encoder values'):        
        sentence_pairs = []
        idx_1 = int(unsupervised_set.iloc[i]['cap_idx']) 
        for j in range(df['cap_idx'].shape[0]):
            idx_2 = int(df.iloc[j]['cap_idx'])
            for x in unsupervised_txt[idx_1]:
                for y in baseline1_txt[idx_2]:
                    sentence_pairs.append([x, y])
        pred = cross_encoder.predict(sentence_pairs, batch_size=256)
        _labels[i] = np.max(pred)
    return _labels

positive_labels = calculate_score(positive_samples)
negative_labels = calculate_score(negative_samples)
neutral_labels = calculate_score(neutral_samples)

_label = unsupervised_set['cap_idx'].copy()
_label['positive_label'] = positive_labels
_label['negative_label'] = negative_labels
_label['neutral_label'] = neutral_labels
