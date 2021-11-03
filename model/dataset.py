import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

join = os.path.join
max_len = 128



def test_train_split(df):
    mask = np.random.rand(len(df)) < 0.8
    train = df[mask]
    test = df[~mask]
    return train, test
    
class dataset(Dataset):
    def __init__(self, data_dirs=['./baseline1', './baseline2'], test=False, random_idx=False):
        self.roots = data_dirs
        self.random_idx = random_idx if not test else False
        self._df = []
        self.video_ids = {}

        for i, root in enumerate(self.roots):
            if test:
                emb_csv = join(root, 'test.csv')
            else:
                emb_csv = join(root, 'train.csv')
            if not os.path.isfile(emb_csv):
                df = pd.read_csv(join(root, '_embedd.csv'), sep='\t')
                train, _test = test_train_split(df)
                train.to_csv(join(root, 'train.csv'), sep='\t')
                _test.to_csv(join(root, 'test.csv'), sep='\t')
            df = pd.read_csv(emb_csv, sep='\t')
            video_ids = df['video_id']
            for id in video_ids:
                self.video_ids[id] = i
            self._df.append(df)
        
        self.caption_data = [np.load(join(root, 'caption_embedd.npy'), allow_pickle=True) for root in self.roots]
        self.title_data = [np.load(join(root, 'title_embedd.npy'), allow_pickle=True) for root in self.roots]
        self.comment_data = [np.load(join(root, 'comments_embedd.npy'), allow_pickle=True) for root in self.roots]
    
    def labels(self):
        label = []
        for df in self._df:
            label += df['label'].to_list()
        return label

    def embedding_size(self):
        return self.caption_data[0][0][0].shape[-1]

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = list(self.video_ids.keys())[idx]
        df_idx = self.video_ids[video_id]
        df = self._df[df_idx]
        video_df = df[video_id == df['video_id']]
        if len(video_df) != 1:
            return self.__getitem__(idx+1)
        
        cap_idx = int(video_df['cap_idx'])
        caption = self.caption_data[df_idx][cap_idx]
        title = np.array(self.title_data[df_idx][cap_idx], dtype=np.float32)

        com_start_idx = int(video_df['com_start_idx'])
        com_end_idx = int(video_df['com_end_idx'])
        if com_start_idx < com_end_idx:
            dif = com_end_idx - com_start_idx
            if self.random_idx and dif > 10:
                sample_comment = self.comment_data[df_idx][com_start_idx: com_end_idx]
                rand_idx = np.random.rand(sample_comment.shape[0]) < 0.9
                sample_comment = sample_comment[rand_idx]
                comments = np.concatenate(sample_comment, axis=0, dtype=np.float32)    

            else:                
                comments = np.concatenate(self.comment_data[df_idx][com_start_idx: com_end_idx], axis=0)    

        elif com_start_idx == com_end_idx:
            comments = self.comment_data[df_idx][com_start_idx-1]
 
        data = np.concatenate([title, caption, comments], axis=0, dtype=np.float32)

        label_raw = int(video_df['label'])

        if label_raw == -1:
            label = 0
        else:
            label = label_raw
        
        tdata = torch.zeros(max_len, self.embedding_size())
        for i in range(min(max_len, data.shape[0])):
            tdata[i] = torch.tensor(data[i], dtype=torch.float32)


        return tdata, torch.tensor(label, dtype=torch.long)


def weighted_sampler(dataset):    
    labels = dataset.labels()
    class_counts = [labels.count(0) +  labels.count(-1), labels.count(1)]
    num_samples = len(labels)
    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    return sampler



