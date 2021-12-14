
import enum
import torch
print("GPU Exists", torch.cuda.is_available())

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import string
from sklearn.metrics.pairwise import cosine_similarity

import re
from emoji import demojize
join = os.path.join

def summarize_text(txt):
    from transformers import BartTokenizer,BartTokenizerFast, BartForConditionalGeneration, BartConfig
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').cuda()
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')
    chunks = chunk_txt(txt)    
    summary = ''
    token = tokenizer(chunks, padding='max_length', truncation=True, return_tensors='pt').to('cuda:0')
    summary_ids = model.generate(token['input_ids'],num_beams=4, max_length=500,  min_length=50, early_stopping=True )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return clean_txt(summary)


def chunk_txt(txt):
    chunk_size = 256
    lst = txt.split(' ')
    chunks = [' '.join(lst[x: x+chunk_size]) for x in range(0, len(lst), chunk_size)]
    return chunks


def clean_txt(txt):
    txt = str(txt)
    txt = demojize(txt)
    # txt = txt.lower()
    txt = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " ", str(txt))
    txt = re.sub(r"<a[^>]*>(.*?)</a>", "", txt)
    
    txt = re.sub(r"<.*?>", "", txt)
    txt = " ".join(txt.split())
    txt = re.sub(r"[^A-Za-z0-9\s]+", "", txt)
    return txt

def scrubbed_captions():
    root = 'scrubbed'
    if not os.path.isdir(f'./{root}'):
        os.makedirs(f'./{root}', exist_ok=True)
    video_df = pd.read_csv('./tmp/video_id.csv', usecols=['video_id', 'title', 'channel_id'])
    # SETUP GLOBAL CSV
    if not os.path.isfile(f'./{root}/_embedd.csv'):
        df = pd.DataFrame(columns=('video_id', 'channel_id', 'title_idx', 'cap_idx', 'com_start_idx', 'com_end_idx'))
        # df = pd.DataFrame(columns=('video_id', 'channel_id', 'title_idx', 'cap_idx'))
        df['video_id'] = video_df['video_id']
        df['channel_id'] = video_df['channel_id']
        df['title_idx'] = np.nan
        df['cap_idx'] = np.nan
        df['com_indexs'] = df.apply(lambda x: '', axis=1)
        df.to_csv(f'./{root}/_embedd.csv', sep='\t')
    else:
        df = pd.read_csv(f'./{root}/_embedd.csv', sep='\t')

    fields = ['video_id', 'captions']
    caption_df = pd.read_csv('./tmp/caption_subset.csv', usecols=fields, low_memory=True)    
    if not os.path.isfile(f"./{root}/caption_text.npy"):        
        tqdm.pandas(desc='caption chunking text')
        caption_df['captions'] = caption_df['captions'].progress_apply(chunk_txt)
        np.save(f"./{root}/caption_text.npy", caption_df['captions'].to_numpy(), allow_pickle=True)
     
    pbar = tqdm(total=caption_df.shape[0], desc='caption updating embedding')
    caption_groups = caption_df.groupby('video_id', as_index=False)    

    for name, group in caption_groups:
        df.loc[df['video_id'] == name, 'cap_idx'] = ','.join([str(x) for x in group.index.tolist()])
        pbar.update(group.shape[0])
    pbar.close()
    df.to_csv(f'./{root}/_embedd.csv', sep='\t')


def scrubbed_comment():
    root = 'scrubbed'
    if not os.path.isdir(f'./{root}'):
        os.makedirs(f'./{root}', exist_ok=True)

    video_df = pd.read_csv('./tmp/video_id.csv')[['video_id', 'title', 'channel_id']]
    # SETUP GLOBAL CSV
    if not os.path.isfile(f'./{root}/_embedd.csv'):
        df = pd.DataFrame(columns=('video_id', 'channel_id', 'title_idx', 'cap_idx', 'com_start_idx', 'com_end_idx'))
        # df = pd.DataFrame(columns=('video_id', 'channel_id', 'title_idx', 'cap_idx'))
        df['video_id'] = video_df['video_id']
        df['channel_id'] = video_df['channel_id']
        df['title_idx'] = np.nan
        df['cap_idx'] = np.nan
        df['com_index'] = df.apply(lambda x: '', axis=1)
        df.to_csv(f'./{root}/_embedd.csv', sep='\t')
    else:
        df = pd.read_csv(f'./{root}/_embedd.csv', sep='\t')
       
    fields=['text', 'video_id']

    if not os.path.isfile(f'./{root}/clean_comment.csv') and not os.path.isfile(f"./{root}/comments_text.npy"):        
        comment_df = pd.read_csv(f'./tmp/comment_subset.csv', sep='\t', usecols=fields, low_memory = True)        
        tqdm.pandas(desc='comment cleanning text')
        comment_df['text'] = comment_df['text'].progress_apply(clean_txt)
        comment_df.to_csv(f'./{root}/clean_comment.csv', sep='\t')
    else:
        comment_df = pd.read_csv(f'./{root}/clean_comment.csv', sep='\t', usecols=fields, low_memory = True)
    

    if not os.path.isfile(f"./{root}/comments_text.npy") :
        tqdm.pandas(desc='comment processing text')
        comment_df['text'] = comment_df['text'].progress_apply(lambda x: str(x).split(string.punctuation))
        np.save(f"./{root}/comments_text.npy", comment_df['text'].to_numpy(), allow_pickle=True)
 
 
    pbar = tqdm(total=comment_df.shape[0], desc='comment updating embedding csv')
    comment_groups = comment_df.groupby('video_id', as_index=False)
    for name, group in comment_groups:
        df.loc[df['video_id'] == name, 'com_index'] = ','.join([str(x) for x in group.index.tolist()])
        pbar.update(group.shape[0])
    
    pbar.close()
    df.to_csv(f'./{root}/_embedd.csv', sep='\t')


def baseline1_captions():
    root = 'baseline1'
    label_id = 'normalized_annotation'   

    if not os.path.isdir(f'./{root}'):
        os.makedirs(f'./{root}', exist_ok=True)
    video_df = pd.read_csv("./tmp/baseline1_videos.csv", usecols=['Captions', 'Video_ID', label_id])
    if not os.path.isfile(f'./{root}/_embedd.csv'):
        df = pd.DataFrame(columns=('video_id', 'channel_id', 'title_idx', 'cap_idx', 'com_start_idx', 'com_end_idx'))
        # df = pd.DataFrame(columns=('video_id', 'channel_id', 'title_idx', 'cap_idx'))
        df['video_id'] = video_df['Video_ID']
        df['channel_id'] = np.nan
        df['title_idx'] = np.nan
        df['cap_idx'] = np.nan
        df['com_indexs'] = df.apply(lambda x: '', axis=1)
        df['label'] = video_df[label_id]
        df.to_csv(f'./{root}/_embedd.csv', sep='\t')
    else:
        df = pd.read_csv(f'./{root}/_embedd.csv', sep='\t')

    if not os.path.isfile(f"./{root}/caption_text.npy"):
        tqdm.pandas(desc="caption chunking text")
        video_df['captions'] = video_df['Captions'].progress_apply(clean_txt)
        video_df['captions'] = video_df['captions'].progress_apply(chunk_txt)
        np.save(f'./{root}/caption_text.npy', video_df['captions'].to_numpy(), allow_pickle=True)
    pbar = tqdm(total=video_df.shape[0], desc='caption updating embedding')
    caption_groups = video_df.groupby('Video_ID', as_index=False)    
    for name, group in caption_groups:
        df.loc[df['video_id'] == name, 'cap_idx'] = ','.join([str(x) for x in group.index.tolist()])
        pbar.update(group.shape[0])
    pbar.close()
    df.to_csv(f'./{root}/_embedd.csv', sep='\t')

def baseline2_comments():
    root = 'baseline2'
    if not os.path.isdir(f'./{root}'):
        os.makedirs(f'./{root}', exist_ok=True)
    comment_df = pd.read_csv('./tmp/baseline2_comments.csv', usecols=['video_id', 'comment'], sep='\t', low_memory=True)
    if not os.path.isfile(f'./{root}/_embedd.csv'):
        df = pd.DataFrame(columns=('video_id', 'channel_id', 'title_idx', 'cap_idx', 'com_start_idx', 'com_end_idx'))
        # df = pd.DataFrame(columns=('video_id', 'channel_id', 'title_idx', 'cap_idx'))
        df['video_id'] = comment_df['video_id']
        df['channel_id'] = np.nan
        df['title_idx'] = np.nan
        df['cap_idx'] = np.nan
        df['com_indexs'] = df.apply(lambda x: '', axis=1)
        df.to_csv(f'./{root}/_embedd.csv', sep='\t')
    else:
        df = pd.read_csv(f'./{root}/_embedd.csv', sep='\t')        

    fields=['text', 'video_id']

    if not os.path.isfile(f'./{root}/clean_comment.csv') and not os.path.isfile(f"./{root}/comments_text.npy"):        
        tqdm.pandas(desc='comment cleanning text')
        comment_df['text'] = comment_df['comment'].progress_apply(clean_txt)
        comment_df.to_csv(f'./{root}/clean_comment.csv', sep='\t')
    elif not os.path.isfile(f"./{root}/comments_text.csv"):
        comment_df = pd.read_csv(f'./{root}/clean_comment.csv', sep='\t', usecols=fields, low_memory = True)
    

    if not os.path.isfile(f"./{root}/comments_text.npy") :
        tqdm.pandas(desc='comment processing text')
        comment_df['text'] = comment_df['text'].progress_apply(lambda x: str(x).split(string.punctuation))
        np.save(f"./{root}/comments_text.npy", comment_df['text'].to_numpy(), allow_pickle=True)
  
    pbar = tqdm(total=comment_df.shape[0], desc='comment updating embedding csv')
    comment_groups = comment_df.groupby('video_id', as_index=False)
    for name, group in comment_groups:
        df.loc[df['video_id'] == name, 'com_index'] = ','.join([str(x) for x in group.index.tolist()])
        pbar.update(group.shape[0])
        
    pbar.close()
    df.to_csv(f'./{root}/_embedd.csv', sep='\t')

    
def convert_to_embedding(root='scrubbed', file_name='comments_text'):
    from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device='cuda')
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cuda")
    batch_size = 512
    lst = np.load(f"./{root}/{file_name}.npy", allow_pickle=True)
    sequence_lengths = np.array([len(l) for l in lst])
    embeddings = [[None for _ in range(l)] for l in sequence_lengths]
    max_sequence_length = sequence_lengths.max()
    print(max_sequence_length)
    for i in range(1, max_sequence_length+1):
        sample = lst[sequence_lengths >= i].tolist()
        sample = [s[i-1] for s in sample]
        output = model.encode(sample, batch_size = batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
        for k,j in enumerate(np.where(sequence_lengths >= i)[0]):
            embeddings[j][i-1] = output[k]
    return np.array(embeddings, copy=False, dtype=object)

def cos_sim(dataset1, dataset2):
    cs = np.zeros([dataset1.shape[0], dataset2.shape[0]])
    sequence_length_1 = np.array([len(dataset1[i]) for i in range(dataset1.shape[0])])
    sequence_length_2 = np.array([len(dataset2[i]) for i in range(dataset2.shape[0])])
    max_sequence_length_1 = sequence_length_1.max()
    max_sequence_length_2 = sequence_length_2.max()
    for i in tqdm(range(1, max_sequence_length_1 + 1), desc='cosine similarity'):
        query_1 = np.where(sequence_length_1 >= i)
        sample_1 = np.array( [s[i-1] for s in dataset1[query_1]])
        for j in range(1, max_sequence_length_2 + 1):
            query_2 = np.where(sequence_length_2 >= j)
            sample_2 = np.array([s[j-1] for s in dataset2[query_2]])
            _cos_sim = cosine_similarity(sample_1, sample_2)            
            for x, k in enumerate(query_1):
                for y, l in enumerate(query_2):
                    cs[k][l] += _cos_sim[x][y]

    return cs

if __name__ == '__main__':
    if not os.path.isfile(f"./baseline1/caption_embedding.npy"):
        baseline1_captions()
        caption_embeddings = convert_to_embedding("baseline1", "caption_text")
        np.save(f"./baseline1/caption_embedding.npy", caption_embeddings, allow_pickle=True)
    
    if not os.path.isfile(f"./baseline2/comment_embeddings.npy"):
        baseline2_comments()
        comment_embeddings = convert_to_embedding('baseline2', 'comments_text') 
        np.save(f'./baseline2/comment_embeddings.npy', comment_embeddings, allow_pickle=True)

    if not os.path.isfile(f"./scrubbed/caption_embedding.npy"):
        scrubbed_captions()
        caption_embeddings = convert_to_embedding('scrubbed', 'caption_text')
        np.save(f"./scrubbed/caption_embedding.npy", caption_embeddings, allow_pickle=True)

    if not os.path.isfile(f"./scrubbed/comment_embeddings.npy"):
        # scrubbed_comment()
        comment_embeddings = convert_to_embedding('scrubbed', 'comments_text')
        np.save(f"./scrubbed/comment_embeddings.npy", comment_embeddings, allow_pickle=True)
   

    # scrubbed_caption_embeddings = np.load(f"./scrubbed/caption_embedding.npy", allow_pickle=True)
    # basline_1_captions_embeddings = np.load(f"./baseline1/caption_embedding.npy", allow_pickle=True)
    # caption_similarity = cos_sim(scrubbed_caption_embeddings, basline_1_captions_embeddings)
    
    # np.save(f"./scrubbed/caption_similairty.npy", caption_similarity)

    # scrubbed_comment_embeddings = np.load(f"./scrubbed/comment_embedding.npy", allow_pickle=True)
    # basleine_2_comment_embedings = np.load(f"./baseline2/comment_embeddings.npy", allow_pickle=True)
    # comment_similarity = cos_sim(scrubbed_comment_embeddings, basleine_2_comment_embedings)
    # np.save(f"./scrubbed/comment_similarity.npy", comment_similarity)
