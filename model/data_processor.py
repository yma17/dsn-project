
import torch
print(torch.cuda.is_available())

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    pipeline,
)
import string

import re
from emoji import demojize
join = os.path.join


model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device='cuda')
nlp_summarizer = pipeline(
    "summarization", 
    model="facebook/bart-large-cnn", 
    tokenizer="facebook/bart-large-cnn", 
    framework="pt", device=0,
)

def convert_to_tokens(txt, chunk=True):
    if chunk or '.' not in txt:
        # sentences = chunk_txt(txt)
        summary = nlp_summarizer(txt)
        sentences = []
        for chunk in summary:
            summary+=chunk['summary_text'].split('.')
    else:
        sentences = txt.split(string.punctuation)
    return model.encode(sentences)

def clean_txt(txt):
    txt = demojize(txt)
    txt = txt.lower()
    txt = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " ", str(txt))
    txt = re.sub(r"<a[^>]*>(.*?)</a>", "", txt)
    
    txt = re.sub(r"<.*?>", "", txt)
    txt = " ".join(txt.split())
    txt = re.sub(r"[^A-Za-z0-9\s]+", "", txt)
    return txt

def chunk_txt(txt):
    chunk_size = 1024
    lst = txt.split(' ')
    chunks = [' '.join(lst[x: x+chunk_size]) for x in range(0, len(lst), chunk_size)]
    return chunks


def scrubbed_noncomment():
    root = 'scrubbed'
    if not os.path.isdir(f'./{root}'):
        os.makedirs(f'./{root}', exist_ok=True)
    video_df = pd.read_csv('./tmp/video_id.csv')[['video_id', 'title', 'channel_id']]
    
    if not os.path.isfile(f'./{root}/_embedd_noncomment.csv'):
        #df = pd.DataFrame(columns=('video_id', 'channel_id', 'title_idx', 'cap_idx', 'com_start_idx', 'com_end_idx'))
        df = pd.DataFrame(columns=('video_id', 'channel_id', 'title_idx', 'cap_idx'))
        df['video_id'] = video_df['video_id']
        df['channel_id'] = video_df['channel_id']
        df['title_idx'] = np.nan
        df['cap_idx'] = np.nan
        #df['com_start_idx'] = np.nan
        #df['com_end_idx'] = np.nan 
    else:
        df = pd.read_csv(f'./{root}/_embedd_noncomment.csv', sep='\t')
   
    caption_df = pd.read_csv('./tmp/caption.csv')[['video_id', 'captions']].to_numpy()
    caption_idx = 0
    title_idx = 0
    #comment_idx = 0

    #embeddings = {'captions':[], 'titles':[], 'comments':[]}
    embeddings = {'captions':[], 'titles':[]}
    if os.path.isfile(f"./{root}/caption_embedd.npy"):
        embeddings['captions'] = np.load(f"./{root}/caption_embedd.npy", allow_pickle=True).tolist()
        caption_idx = len(embeddings['captions'])
    if os.path.isfile(f"./{root}/title_embedd.npy"):
        embeddings['titles'] = np.load(f"./{root}/title_embedd.npy", allow_pickle=True).tolist()
        title_idx = len(embeddings['titles'])
    if os.path.isfile(f"./{root}/comments_embedd.npy"):
        embeddings['comments'] = np.load(f"./{root}/comments_embedd.npy", allow_pickle=True).tolist()
    
    pbar = tqdm(total=caption_df.shape[0], desc='embedding')

    for video_id, caption in caption_df:
        msk = df['video_id'] == video_id
        embeddings['captions'] = convert_to_tokens(caption)
        df.loc[msk, 'cap_idx'] = int(caption_idx)  
        caption_idx += 1
        title = video_df.loc[video_id == video_df['video_id']]['title']            
        embeddings['titles'] = convert_to_tokens(clean_txt(str(title)), False)
        df.loc[msk, 'title_idx'] = int(caption_idx)                  
        title_idx += 1
        pbar.update(1)
    pbar.close()

    df.to_csv(f'./{root}/_embedd_noncomment.csv', sep='\t')

    np.save(f"./{root}/caption_embedd.npy", np.array(embeddings['captions'], dtype=object))
    np.save(f"./{root}/title_embedd.npy", np.array(embeddings['titles'], dtype=object))
    #np.save(f"./{root}/comments_embedd.npy", np.array(embeddings['comments'], dtype=object))

def scrubbed_comment():
    root = 'scrubbed'
    if not os.path.isdir(f'./{root}'):
        os.makedirs(f'./{root}', exist_ok=True)

    #comment_df = pd.read_csv(f'./{root}/comment_audit.csv', sep='\t')
    comment_df = pd.read_csv(f'./{root}/comment_sample.csv', sep='\t')
    comment_df = comment_df[['text', 'video_id']]

    if not os.path.isfile(f'./{root}/_embedd_comment.csv'):
        df = pd.DataFrame(columns=('video_id'))
        df['video_id'] = comment_df['video_id']
    else:
        df = pd.read_csv(f'./{root}/_embedd_comment.csv', sep='\t')

    embeddings = {'comments': []}
    if os.path.isfile(f"./{root}/comments_embedd.npy"):
        embeddings['comments'] = np.load(f"./{root}/comments_embedd.npy", allow_pickle=True).tolist()
    
    pbar = tqdm(total=comment_df.shape[0], desc='embedding')

    comment_df_np = comment_df[['video_id', 'text']].to_numpy()
    for _, text in comment_df_np:
        comment_embedd = convert_to_tokens(clean_txt(str(text)), True)
        embeddings['comments'].append(comment_embedd)
        pbar.update(1)
    pbar.close()

    df.to_csv(f'./{root}/_embedd_comment.csv', sep='\t')
    np.save(f"./{root}/comments_embedd.npy", np.array(embeddings['comments'], dtype=object))

def scrubbed_combined():
    # TODO - combine embedd_noncomment and embedd_comment
    # TODO - compute com_start_idx and com_end_idx
    pass

def baseline1():
    baseline = 'baseline1'
    if not os.path.isdir(f'./{baseline}'):
        os.makedirs(f'./{baseline}', exist_ok=True)

    label_id = 'normalized_annotation' if baseline == 'baseline1' else 'misinformation'        
    
    comment_df = pd.read_csv(f'./tmp/{baseline}_comments.csv', sep='\t')
    video_df = pd.read_csv(f'./tmp/{baseline}_videos.csv', sep='\t')[['video_id', 'title', 'captions', label_id]].to_numpy()
    pbar = tqdm(total=video_df.shape[0])
    _embedding = {'video_id': [], 'cap_idx': [], 'com_end_idx':[], 'com_start_idx':[], 'label':[], }
    comment_idx = 0
    caption_idx = 0
    
    embeddings = {"captions":[], "titles":[], "comments":[]}
    
    for video_id, video_title, captions, label in video_df:
        caption_embedd = convert_to_tokens(captions.strip('>'))
        title_embedd = convert_to_tokens(clean_txt(video_title), False)

        embeddings['captions'].append(caption_embedd)
        embeddings['titles'].append(title_embedd)

        _embedding['cap_idx'].append(caption_idx)
        _embedding['video_id'].append(video_id)
        _embedding['label'].append(int(label))
        _embedding['com_start_idx'].append(comment_idx)
        comment_seq = comment_df.loc[comment_df['video_id'] == video_id]['comment']
        for comment in comment_seq:
            try:                
                comment_embedd = convert_to_tokens(clean_txt(str(comment)), False)
                comment_idx += 1
                embeddings['comments'].append(comment_embedd)
            except Exception as e:
                print(e, clean_txt(comment))

        _embedding['com_end_idx'].append(comment_idx)
        caption_idx += 1
        pbar.update(1)
    pbar.close()
        
    pd.DataFrame(_embedding).to_csv(f'./{baseline}/_embedd.csv', sep='\t')

    np.save(f"./{baseline}/caption_embedd.npy", np.array(embeddings['captions'], dtype=object))
    np.save(f"./{baseline}/title_embedd.npy", np.array(embeddings['titles'], dtype=object))
    np.save(f"./{baseline}/comments_embedd.npy", np.array(embeddings['comments'], dtype=object))
    print(len(embeddings['captions']), len(embeddings['titles'], embeddings['comments']))

def baseline2():
    baseline = 'baseline2'
    if not os.path.isdir(f'./{baseline}'):
        os.makedirs(f'./{baseline}', exist_ok=True)

    label_id = 'normalized_annotation' if baseline == 'baseline1' else 'misinformation'        
    
    comment_df = pd.read_csv(f'./tmp/{baseline}_comments.csv', sep='\t')
    video_df = pd.read_csv(f'./tmp/{baseline}_videos.csv', sep='\t')[['video_id', 'title', 'captions', label_id]].to_numpy()
    pbar = tqdm(total=video_df.shape[0])
    _embedding = {'video_id': [], 'cap_idx': [], 'com_end_idx':[], 'com_start_idx':[], 'label':[], }
    comment_idx = 0
    caption_idx = 0
    
    embeddings = {"captions":[], "titles":[], "comments":[]}
    
    for video_id, video_title, captions, label in video_df:
        caption_embedd = convert_to_tokens(captions)
        title_embedd = convert_to_tokens(clean_txt(video_title), False)

        embeddings['captions'].append(caption_embedd)
        embeddings['titles'].append(title_embedd)

        _embedding['cap_idx'].append(caption_idx)
        _embedding['video_id'].append(video_id)
        _embedding['label'].append(int(label))
        _embedding['com_start_idx'].append(comment_idx)
        comment_seq = comment_df.loc[comment_df['video_id'] == video_id]['comment']
        for comment in comment_seq:
            try:                
                comment_embedd = convert_to_tokens(clean_txt(str(comment)), False)
                comment_idx += 1
                embeddings['comments'].append(comment_embedd)
            except Exception as e:
                print(e, clean_txt(comment))

        _embedding['com_end_idx'].append(comment_idx)
        caption_idx += 1
        pbar.update(1)
    pbar.close()
        
    pd.DataFrame(_embedding).to_csv(f'./{baseline}/_embedd.csv', sep='\t')

    np.save(f"./{baseline}/caption_embedd.npy", np.array(embeddings['captions'], dtype=object))
    np.save(f"./{baseline}/title_embedd.npy", np.array(embeddings['titles'], dtype=object))
    np.save(f"./{baseline}/comments_embedd.npy", np.array(embeddings['comments'], dtype=object))
    print(len(embeddings['captions']), len(embeddings['titles'], embeddings['comments']))

if __name__ == '__main__':
    job = sys.argv[1]
    if job == 'scrubbed_noncomment':
        scrubbed_noncomment()
    elif job == 'scrubbed_comment':
        scrubbed_comment()
    #scrubbed_combined()
