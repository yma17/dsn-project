from collections import Counter
import numpy as np

def print_comment_statistics(comment_df):
    print("\nPrinting comment statistics:")
    print("Columns:", comment_df.columns)
    print("Counting null values:", comment_df.isna().sum())
    print("Total comments:", len(comment_df))
    print("Earliest comment datetime:", comment_df['datetime'].min())
    print("Latest comment datetime:", comment_df['datetime'].max())
    print("Comment distribution by year:")
    z = [s[:4] for s in comment_df['datetime'].to_list()]
    print(Counter(z))
    print("Comment distribution by likes:")
    z = np.array([int(i) for i in comment_df['likes'].to_list()])
    print("min,max,avg,std", z.min(), z.max(), z.mean(), z.std())
    print("Comment distribution by viewer rating:")
    z = [b for b in comment_df['viewer_rating'].to_list()]
    print(Counter(z))
    print("Comment distribution by number of tokens:")
    z = np.array([len(s.split(' ')) for s in comment_df['text']])
    print("min,max,avg,std", z.min(), z.max(), z.mean(), z.std())
    print("Number of unique videos:", len(set(comment_df['video_id'].to_list())))
    print("Num comments distribution by video:")
    z = comment_df.groupby(['video_id']).size().to_numpy()
    print("min,max,avg,std", z.min(), z.max(), z.mean(), z.std())


def print_caption_statistics(caption_df):
    print("\nPrinting caption statistics:")
    print("Columns:", caption_df.columns)
    print("Counting null values:", caption_df.isna().sum())
    print("Caption distribution by number of tokens:")
    z = np.array([len(s.split(' ')) for s in caption_df['captions']])
    print("min,max,avg,std", z.min(), z.max(), z.mean(), z.std())