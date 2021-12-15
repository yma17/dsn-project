import numpy as np
import pandas as pd
from sentence_transformers import models, losses, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
root = '.'
bi_model_name = 'all-distilroberta-v1'
bi_encoder_path = f'{root}/output/bi_{bi_model_name}/'
bi_encoder_path = f'{root}/output/bi_out'

bi_encoder = SentenceTransformer(bi_encoder_path)
# for x in bi_encoder.named_modules():
#     print(x)
cap_idxs = pd.read_csv('./baseline1/test.csv', sep='\t', usecols=['cap_idx', 'video_id'])
cap_txt = np.load('./baseline1/caption_text.npy', allow_pickle=True)
baseline1 = pd.read_csv('./tmp/baseline1_videos.csv', usecols=['normalized_annotation', 'Video_ID'])

def convert_to_embedding(lst, model):
    from sentence_transformers import SentenceTransformer
    batch_size = 32
    sequence_lengths = np.array([len(l) for l in lst])
    embeddings = [[None for _ in range(l)] for l in sequence_lengths]
    max_sequence_length = sequence_lengths.max()
    print(max_sequence_length)
    for i in trange(1, max_sequence_length+1, desc='embedding'):
        sample = lst[sequence_lengths >= i].tolist()
        sample = [s[i-1] for s in sample]
        output = model.encode(sample, batch_size = batch_size, convert_to_numpy=True)
        for k,j in enumerate(np.where(sequence_lengths >= i)[0]):
            embeddings[j][i-1] = output[k]
    
    model_embedding = []
    for i in range(len(embeddings)):
        seq = np.array(embeddings[i])

        model_embedding.append(seq.mean(0))

    return np.array(model_embedding, copy=False), np.array(embeddings, dtype=object)

_order1 = baseline1['Video_ID'].to_numpy()
_order2 = cap_idxs['video_id'].to_numpy()
differ = np.where(_order1 != _order2)[0]
print(differ, len(differ))

embedd, _embedd = convert_to_embedding(cap_txt, bi_encoder)
print(_order1.shape, _order2.shape, embedd.shape)

for d in range(len(differ)):
    correct_value = _order1[d]
    idx = np.where(_order2 == correct_value)[0][0]
    _order2[d], _order2[idx] = _order2[idx], _order2[d]
    embedd[d], embedd[idx] = embedd[idx], embedd[d]
print(embedd.shape)
np.save('./baseline1/bi_embedding.npy', embedd, )
np.save('./baseline1/bi2_embedding.npy', _embedd, allow_pickle=True)

# from numpy.linalg import norm
# cos_sim = np.matmul(embedd @ embedd.T / (norm(embedd) ** 2))
# print("CS" )
