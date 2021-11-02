import os
import math
import torch
from torch._C import device
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

join = os.path.join
max_len = 256

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
                train, test = test_train_split(df)
                train.to_csv(join(root, 'train.csv'), sep='\t')
                test.to_csv(join(root, 'test.csv'), sep='\t')
            df = pd.read_csv(emb_csv, sep='\t')
            video_ids = df['video_id']
            for id in video_ids:
                self.video_ids[id] = i
            self._df.append(df)
    
        
        
        self.caption_data = [np.load(join(root, 'caption_embedd.npy'), allow_pickle=True) for root in self.roots]
        self.title_data = [np.load(join(root, 'title_embedd.npy'), allow_pickle=True) for root in self.roots]
        self.comment_data = [np.load(join(root, 'comments_embedd.npy'), allow_pickle=True) for root in self.roots]
    
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
            label = 2
        else:
            label = label_raw
        
        tdata = torch.zeros(max_len, self.embedding_size())
        for i in range(min(max_len, data.shape[0])):
            tdata[i] = torch.tensor(data[i], dtype=torch.float32)


        return tdata, torch.tensor(label, dtype=torch.long)



class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """        
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class LSTM(nn.Module):

    def __init__(self, embedding_size, dimension=1024, n_layers=8):
        super(LSTM, self).__init__()
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=dimension,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 3)

    def forward(self, text, text_len):
        output, _ = self.lstm(text)
        
        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        return text_fea

if __name__ == "__main__":
    data_set = dataset(random_idx=True)
    test_set = dataset(test=True)

    data_loader = DataLoader(data_set, batch_size=16, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=16, drop_last=True)

    n_epochs = 100
    eval_every = 30
    embedding_size = 768
    nTokens = 2
    model = LSTM(embedding_size, 1024)
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    test_criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    global_step = 0

    for epoch in range(n_epochs):
        running_loss = 0
        valid_running_loss = 0

        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            logit = model(x, max_len)
            loss = criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item()

            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    for _x, _y in test_loader:
                        _x = _x.to(device)
                        _y = _y.to(device)
                        _logit = model(x, max_len)
                        _loss = test_criterion(_logit, _y)
                        valid_running_loss += _loss.item()
                model.train()

        average_train_loss = running_loss / eval_every
        average_valid_loss = valid_running_loss / len(test_loader) 
        print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                    .format(epoch+1, n_epochs, global_step, n_epochs*len(data_loader),
                            average_train_loss, average_valid_loss))