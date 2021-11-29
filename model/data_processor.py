
import torch
print("GPU Exists", torch.cuda.is_available())

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import string

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

def convert_to_tokens(txt):
    txt = clean_txt(txt)
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    sentences = txt.split(string.punctuation)
    return model.encode(sentences)

def clean_txt(txt):
    txt = str(txt)
    txt = demojize(txt)
    txt = txt.lower()
    txt = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " ", str(txt))
    txt = re.sub(r"<a[^>]*>(.*?)</a>", "", txt)
    
    txt = re.sub(r"<.*?>", "", txt)
    txt = " ".join(txt.split())
    txt = re.sub(r"[^A-Za-z0-9\s]+", "", txt)
    return txt

def chunk_txt(txt):
    chunk_size = 512
    lst = txt.split(' ')
    chunks = [' '.join(lst[x: x+chunk_size]) for x in range(0, len(lst), chunk_size)]
    return chunks

def scrubbed_captions():
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
        df['com_start_idx'] = np.nan
        df['com_end_idx'] = np.nan
        df.to_csv(f'./{root}/_embedd.csv', sep='\t')
    else:
        df = pd.read_csv(f'./{root}/_embedd.csv', sep='\t')

    caption_df = pd.read_csv('./tmp/caption.csv')[['video_id', 'captions']]
    caption_df = caption_df.set_index(['video_id'])
    # CAPTION SUMMARIZATION
    

    if not os.path.isfile('./tmp/_caption.csv'):
        pbar = tqdm(total=caption_df.shape[0], desc='summarizing')
        for index, row in caption_df.iterrows():
            caption = row['captions']
            summary = summarize_text(model, tokenizer, caption)
            caption_df.loc[index,'caption_summary'] = summary
            pbar.update(1)
        # caption_df['caption_summary'] = caption_df['captions'].apply(summarize_text)
        caption_df.to_csv('./tmp/_caption.csv', sep='\t')
        pbar.close()
    else:
        caption_df = pd.read_csv('./tmp/_caption.csv', sep='\t')

    caption_df = caption_df.to_numpy()    
    caption_idx = 0
    title_idx = 0

    # CAPTION EMBEDDING
    embeddings = {'captions': [], 'titles': []}
    if os.path.isfile(f"./{root}/caption_embedd.npy"):
        embeddings['captions'] = np.load(f"./{root}/caption_embedd.npy", allow_pickle=True).tolist()
        caption_idx = len(embeddings['captions'])
    if os.path.isfile(f"./{root}/title_embedd.npy"):
        embeddings['titles'] = np.load(f"./{root}/title_embedd.npy", allow_pickle=True).tolist()
        title_idx = len(embeddings['titles'])

    pbar = tqdm(total=caption_df.shape[0], desc='embedding')
    del model
    del tokenizer
    # model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device='cuda')

    for video_id, caption, caption_summary in caption_df:
        msk = df['video_id'] == video_id
        embeddings['captions'] = convert_to_tokens(model, caption_summary)
        df.loc[msk, 'cap_idx'] = int(caption_idx)  
        caption_idx += 1
        title = video_df.loc[video_id == video_df['video_id']]['title']            
        embeddings['titles'] = convert_to_tokens(model, clean_txt(str(title)))
        df.loc[msk, 'title_idx'] = int(caption_idx)                  
        title_idx += 1
        pbar.update(1)

    pbar.close()

    df.to_csv(f'./{root}/_embedd.csv', sep='\t')

    np.save(f"./{root}/caption_embedd.npy", np.array(embeddings['captions'], dtype=object))
    np.save(f"./{root}/title_embedd.npy", np.array(embeddings['titles'], dtype=object))


def clean_txt(txt):
    txt = str(txt)
    txt = demojize(txt)
    txt = txt.lower()
    txt = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " ", str(txt))
    txt = re.sub(r"<a[^>]*>(.*?)</a>", "", txt)
    
    txt = re.sub(r"<.*?>", "", txt)
    txt = " ".join(txt.split())
    txt = re.sub(r"[^A-Za-z0-9\s]+", "", txt)
    return txt


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
        df['com_start_idx'] = np.nan
        df['com_end_idx'] = np.nan
        df.to_csv(f'./{root}/_embedd.csv', sep='\t')
    else:
        df = pd.read_csv(f'./{root}/_embedd.csv', sep='\t')
    fields=['text', 'video_id']
    if not os.path.isfile(f'./tmp/clean_comment_audit.csv'):        
        comment_df = pd.read_csv(f'./tmp/comment_audit.csv', sep='\t', usecols=fields, low_memory = True)        
        tqdm.pandas(desc='cleanning text')
        comment_df['text'] = comment_df['text'].progress_apply(clean_txt)
        comment_df.to_csv(f'./tmp/clean_comment_audit.csv', sep='\t')
    else:
        comment_df = pd.read_csv(f'./tmp/clean_comment_audit.csv', sep='\t', usecols=fields, low_memory = True)
    

    counter = 0
    if not os.path.isfile(f"./{root}/comments_text.npy"):
        tqdm.pandas(desc='processing text')
        comment_df['text'] = comment_df['text'].progress_apply(lambda x: str(x).split(string.punctuation))
        comment_df = comment_df['text'].progress_apply(pd.Series).melt(id_vars=['video_id'], value_name='text')
        print(comment_df.shape)
        print(comment_df.head())
        comment_df.to_csv("./{root}/processed_comment_text.csv", sep='\t', usecols=fields)
        # np.save(f"./{root}/comments_text.npy", comment_df['text'].to_numpy(), allow_pickle=True).dropna()

    pbar = tqdm(total=comment_df.shape[0], desc='updating embedding csv')
    comment_groups = comment_df.groupby('video_id')
    com_start_idx = []
    com_end_idx = []

    for name, group in comment_groups:
        group = group.to_numpy()
        pbar.update(group.shape[0])
        com_start_idx.append((name, counter))
        counter += group.shape[0]
        com_end_idx.append((name, counter))
        if counter % 1000 == 0:
            df.to_csv(f'./{root}/_embedd.csv', sep='\t')
 
    df['com_start_idx'] = com_start_idx
    df['com_end_idx'] = com_end_idx
      
    pbar.close()
    df.to_csv(f'./{root}/_embedd.csv', sep='\t')


def convert_to_embedding(root='scrubbed'):
    from sentence_transformers import SentenceTransformer
    lst = np.load(f"./{root}/comments_text.npy", allow_pickle=True).tolist()
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device='cuda')
    return model.encode(lst, batch_size=32, show_progress_bar=True, )

if __name__ == '__main__':
    # job = sys.argv[1]
    # if job == 'captions':
    #   scrubbed_captions()
    # else:
    # convert_to_embedding()
    scrubbed_comment()
    #scrubbed_combined()
