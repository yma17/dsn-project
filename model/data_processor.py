import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import re
from emoji import demojize
join = os.path.join

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device='cuda')

def convert_to_tokens(txt, chunk=True):
    if chunk or '.' not in txt:
        sentences = chunk_txt(txt)
    else:
        sentences = txt.split('.')
    return model.encode(sentences)

def clean_txt(txt):
    txt = demojize(txt)
    txt = txt.lower()
    txt = re.sub(r"https?://\S+", "", txt)
    txt = re.sub(r"<a[^>]*>(.*?)</a>", "", txt)
    
    txt = re.sub(r"<.*?>", "", txt)
    txt = " ".join(txt.split())
    txt = re.sub(r"[^A-Za-z0-9\s]+", "", txt)
    return txt

def chunk_txt(txt):
    lst = txt.split(' ')
    chunks = [' '.join(lst[x: x+384]) for x in range(0, len(lst), 384)]
    return chunks


def scrubbed():
    root = 'scrubbed'
    if not os.path.isdir(f'./{root}'):
        os.makedirs(f'./{root}', exist_ok=True)
    video_df = pd.read_csv('./tmp/video_id.csv')[['video_id', 'title', 'channel_id']]
    
    if not os.path.isfile(f'./{root}/_embedd.csv'):
        df = pd.DataFrame(columns=('video_id', 'channel_id', 'title_idx', 'cap_idx', 'com_start_idx', 'com_end_idx'))
        df['video_id'] = video_df['video_id']
        df['channel_id'] = video_df['channel_id']
        df['title_idx'] = np.nan
        df['cap_idx'] = np.nan
        df['com_start_idx'] = np.nan
        df['com_end_idx'] = np.nan 
    else:
        df = pd.read_csv(f'./{root}/_embedd.csv', sep='\t')
    caption_df = pd.read_csv('./tmp/caption_0.csv')[['video_id', 'captions']].to_numpy()
    caption_idx = 0
    title_idx = 0
    comment_idx = 0

    embeddings = {'captions':[], 'titles':[], 'comments':[]}
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
        

    df.to_csv(f'./{root}/_embedd.csv', sep='\t')

    np.save(f"./{root}/caption_embedd.npy", np.array(embeddings['captions'], dtype=object))
    np.save(f"./{root}/title_embedd.npy", np.array(embeddings['titles'], dtype=object))
    np.save(f"./{root}/comments_embedd.npy", np.array(embeddings['comments'], dtype=object))
    
    # _embedding = {'video_id': [], 'cap_idx': [], 'title_idx': []}
    # embeddings = {'captions':[], "title": [], "comments":[]}


    # pbar = tqdm(total=video_df.shape[0], desc='title_embedd')

    # title_idx = 0
    # for video_id, title, channel_id in video_df:
    #     title_idx += 1
    #     title_embedd = convert_to_tokens(clean_txt(title), False)
    #     _embedding['video_id'].append(video_id)
    #     _embedding['title_idx'] = title_idx
    #     embeddings['title'].append(title_embedd)
    #     pbar.update(1)

    # caption_df = pd.read_csv('./tmp/captions_0.csv')['video_id', 'title', 'channel_id']
    # caption_idx = 0
    # pbar_1 = tqdm
    # for video_id, caption in caption_df:
    #     caption_embedd = convert_to_tokens(caption)
    #     _embedding['cap_idx'].append(caption_idx)
    #     embeddings['captions'].append(caption_embedd)
    #     caption_idx += 1

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
   scrubbed()

