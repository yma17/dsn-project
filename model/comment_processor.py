import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
join = os.path.join


TEST_CSV = 'baseline_comments.csv'


class commentProcessor(object):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-xlm-r-multilingual-v1", device='cuda')

    def __call__(self, summary, comment):
        embedd1 = self.model.encode(summary)
        embedd2 = self.model.encode(comment)
        k = util.cos_sim(embedd1, embedd2)
        return k





if __name__ == '__main__':
    comment_df = pd.read_csv('./tmp/baseline1_comments.csv')
    video_df = pd.read_csv('./tmp/baseline1_videos.csv')[['video_id', 'title', 'captions']].to_numpy()
    comment_process = commentProcessor()

    comment_embeddings = {'video_id': [], 'comment_embedd': []}
    caption_embeddings = {'video_id': [], 'caption_embedd': [], 'title_embedd': []}

    for video_id, video_title, captions in video_df:
        caption_embedd = comment_process.model.encode(captions.strip('>'))
        title_embedd = comment_process.model.encode(video_title)
        caption_embeddings['caption_embedd'].append(caption_embedd)
        caption_embeddings['video_id'].append(video_id)
        caption_embeddings['title_embedd'].append(title_embedd)
        
        comment_seq = comment_df.loc[comment_df['video_id'] == video_id]['comment']
        for comment in comment_seq:
            try:
                comment_embedd = comment_process.model.encode(str(comment).split('.'))
                comment_embeddings['video_id'].append(video_id)
                comment_embeddings['comment_embedd'].append(comment_embedd)
            except:
                print(comment)
                raise TypeError("comment is not the right type")

    comment_embedd_df = pd.DataFrame(comment_embeddings)
    caption_embedd_df = pd.DataFrame(caption_embeddings)
    comment_embedd_df.to_csv('./comment_embedd_1.csv')
    caption_embedd_df.to_csv('./caption_embedd_1.csv')


