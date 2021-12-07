import os
import torch
import random
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
    def __init__(self, data_dirs=['./baseline1', './baseline2'], test=False, random_idx=False, id=-1):
        self.roots = data_dirs
        self.random_idx = random_idx if not test else False
        self._df = []
        self.video_ids = {}
        if id == 0 and test:
            self.roots = ['./baseline1']
        if id == 1 and test:
            self.roots = ['./baseline2']
        
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
            video_ids = df['video_id'].to_list()
            for id in video_ids:
                self.video_ids[id] = i
            self._df.append(df)
        self.caption_data = [np.load(join(root, 'caption_embedd.npy'), allow_pickle=True) for root in self.roots]
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


from sentence_transformers.readers import InputExample
class CrossEncoderDataset(Dataset):
    def __init__(self, root='./baseline1', iter=10000):
        self.root = root
        self.iter = iter
        if not os.path.isfile(join(root, 'train.csv')):
            supervised_set = pd.read_csv(f'./{root}/_embedd.csv', usecols=['video_id', 'label', 'cap_idx'], sep='\t')             
            train, _test = test_train_split(supervised_set)
            train.to_csv(join(root, 'train.csv'), sep='\t')
            _test.to_csv(join(root, 'test.csv'), sep='\t')
        else:
            supervised_set = pd.read_csv(join(root, 'train.csv'), usecols=['video_id', 'label', 'cap_idx'], sep='\t')

        self.supervised_groups = supervised_set.groupby(pd.Grouper('label'), as_index=False)
        self.caption_text = np.load(f'./{root}/caption_text.npy', allow_pickle=True)

        self.labels = dict([(i, self.supervised_groups.get_group(i).size) for i in self.supervised_groups.groups])
        self.label = list(self.labels.keys())
        print(self.labels)

    def __len__(self):
        return self.iter
    
    def test(self):
        emb_csv = join(self.root, 'test.csv')
        supervised_set = pd.read_csv(emb_csv, usecols=['video_id', 'label', 'cap_idx'], sep='\t')
        supervised_groups = supervised_set.groupby(pd.Grouper('label'), as_index=False)        
        labels = dict([(i, supervised_groups.get_group(i).size) for i in supervised_groups.groups])
        label = list(labels.keys())        
        sentence_pairs, sim = [], []
        for _ in range(1000):
            idx_1 = random.randint(0, len(label)-1)
            idx_2 = random.randint(0, len(label)-1)
            label_1 = label[idx_1]
            label_2 = label[idx_2]
            group_1 = supervised_groups.get_group(label_1)['cap_idx'].to_numpy()
            group_2 = supervised_groups.get_group(label_2)['cap_idx'].to_numpy()
            group_idx_1 = random.randint(0, group_1.shape[0]-1)
            group_idx_2 = random.randint(0, group_2.shape[0]-1)
            cap_idx_1 = group_1[group_idx_1]
            cap_idx_2 = group_2[group_idx_2]
            try:
                cap_idx_1 = int(cap_idx_1)
            except:
                cap_idx_1 = [int(x) for x in cap_idx_1.split(',')][0]
            try:
                cap_idx_2 = int(cap_idx_2)
            except:
                cap_idx_2 = [int(x) for x in cap_idx_2.split(',')][0]
            text1 = self.caption_text[cap_idx_1]
            text2 = self.caption_text[cap_idx_2]
            group_text_1 = random.randint(0, len(text1)-1)
            group_text_2 = random.randint(0, len(text2)-1)
            _label = 1 if idx_1 == idx_2 else 0            
            sentence_pairs.append(InputExample(texts=[text1[group_text_1], text2[group_text_2]], label=_label))
        return sentence_pairs

    def __getitem__(self, idx):
        idx_1 = random.randint(0, len(self.label)-1)
        idx_2 = random.randint(0, len(self.label)-1)
        label_1 = self.label[idx_1]
        label_2 = self.label[idx_2]
        group_1 = self.supervised_groups.get_group(label_1)['cap_idx'].to_numpy()
        group_2 = self.supervised_groups.get_group(label_2)['cap_idx'].to_numpy()
        group_idx_1 = random.randint(0, group_1.shape[0]-1)
        group_idx_2 = random.randint(0, group_2.shape[0]-1)
        cap_idx_1 = group_1[group_idx_1]
        cap_idx_2 = group_2[group_idx_2]
        try:
            cap_idx_1 = int(cap_idx_1)
        except:
            cap_idx_1 = [int(x) for x in cap_idx_1.split(',')][0]
        try:
            cap_idx_2 = int(cap_idx_2)
        except:
            cap_idx_2 = [int(x) for x in cap_idx_2.split(',')][0]
        text1 = self.caption_text[cap_idx_1]
        text2 = self.caption_text[cap_idx_2]
        group_text_1 = random.randint(0, len(text1)-1)
        group_text_2 = random.randint(0, len(text2)-1)
        label = 1 if idx_1 == idx_2 else 0
        return  InputExample(texts=[text1[group_text_1], text2[group_text_2]], label=label)

class BI_Encoder(Dataset):
    def __init__(self, root='scrubbed'):
        self.root = root
        # supervised_set = pd.read_csv(f'./{root}/_embedd.csv', usecols=['video_id', 'cap_idx'], sep='\t', low_memory=True) 
        self.caption_text = np.load(f'./{root}/caption_text.npy', allow_pickle=True)
        
        
    def __len__(self):
        return len(self.sentence_pair)
    
    def label(self, model):
        root = 'baseline1'
        # supervised_set = pd.read_csv(f'./{root}/_embedd.csv', usecols=['video_id', 'label', 'cap_idx'], sep='\t')             
        baseline1_caption_text = np.load(f'./{root}/caption_text.npy', allow_pickle=True)
        self.sentence_pair = []
        for i in range(self.caption_text.shape[0]):
            text_block = self.caption_text[i]
            idx = random.randint(0, baseline1_caption_text.shape[0]-1)
            baseline_text_block = baseline1_caption_text[idx]
            for t_block in text_block:
                baseline_idx = random.randint(0, len(baseline_text_block)-1)
                t2_block = baseline_text_block[baseline_idx]
                self.sentence_pair.append([t_block, t2_block])
                self.sentence_pair.append([t2_block, t_block])
        self.sim = model.predict(self.sentence_pair, batch_size=128, show_progress_bar=True)
        return self.sim

    def __getitem__(self, idx):
        txt1, txt2 = self.sentence_pair[idx]
        score = self.sim[idx]
        return InputExample(texts=[txt1, txt2], score=score)