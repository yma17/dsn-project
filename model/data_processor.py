import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

import re
from emoji import demojize


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

if __name__ == '__main__':
    comment_df = pd.read_csv('./tmp/baseline1_comments.csv', sep='\t')
    video_df = pd.read_csv('./tmp/baseline1_videos.csv', sep='\t')[['video_id', 'title', 'captions']].to_numpy()
    comment_embeddings = {'video_id': [], 'idx': []}
    caption_embeddings = {'video_id': [], 'idx': []}
    comment_idx = 0
    caption_idx = 0
    
    embeddings = {"captions":[], "titles":[], "comments":[]}
    
    for video_id, video_title, captions in video_df:
        caption_embedd = convert_to_tokens(captions.strip('>'))
        title_embedd = convert_to_tokens(clean_txt(video_title), False)

        embeddings['captions'].append(caption_embedd)
        embeddings['titles'].append(title_embedd)

        caption_embeddings['idx'].append(caption_idx)
        caption_embeddings['video_id'].append(video_id)
        caption_idx += 1

        comment_seq = comment_df.loc[comment_df['video_id'] == video_id]['comment']
        for comment in comment_seq:
            try:
                comment_embedd = convert_to_tokens(clean_txt(str(comment)), False)
                embeddings['comments'].append(comment_embedd)
                comment_embeddings['video_id'].append(video_id)
                comment_embeddings['idx'].append(comment_idx)
                comment_idx += 1
            except Exception as e:
                print(e, clean_txt(comment))
                
       
    comment_embedd_df = pd.DataFrame(comment_embeddings)
    caption_embedd_df = pd.DataFrame(caption_embeddings)
    comment_embedd_df.to_csv('./comment_embedd_1.csv')
    caption_embedd_df.to_csv('./caption_embedd_1\
        .csv')

    captions = np.array(embeddings['captions'], dtype=object)
    titles = np.array(embeddings['titles'], dtype=object)
    comments = np.array(embeddings['comments'], dtype=object)

    np.save("caption_embedd_1.npy", captions)
    np.save("title_embedd_1.npy", titles)
    np.save("comments_embedd_1.npy", comments)

