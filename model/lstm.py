import os
from numpy.lib.function_base import percentile
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import dataset, weighted_sampler, max_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

join = os.path.join

class LSTM(nn.Module):

    def __init__(self, embedding_size, dimension=1024, n_layers=8):
        super(LSTM, self).__init__()
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=dimension,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.2)
        self.fc_1 = nn.Linear(2*dimension, 2*dimension)
        self.fc_act_1 = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(2*dimension, 2)
        self.fc_act = nn.Sigmoid()


    def forward(self, text, text_len):
        output, _ = self.lstm(text)
        
        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        self.fc_act_1(out_reduced)
        text_fea = self.drop(out_reduced)
        text_fea = self.fc(text_fea)
        
        # text_fea = torch.squeeze(text_fea, 1)
        text_fea = self.fc_act(text_fea)
        return text_fea

if __name__ == "__main__":
    data_set = dataset(random_idx=True)
    test_set = dataset(random_idx=True, test=True)
    sampler = weighted_sampler(data_set)
    samper_1 = weighted_sampler(test_set)
    data_loader = DataLoader(data_set, batch_size=32, drop_last=True,  pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, drop_last=True, pin_memory=True, shuffle=True)

    n_epochs = 1000
    eval_every = 50
    embedding_size = 768
    nTokens = 2
    model = LSTM(embedding_size, 1024, n_layers=8)
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    test_criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
            running_loss += loss.mean(0).item()

            if global_step % eval_every == 0:
                model.eval()
                running_f1 = 0
                running_acc = 0
                running_rec = 0
                running_pre = 0
                
                with torch.no_grad():
                    for _x, _y in test_loader:
                        _x = _x.to(device)
                        _y = _y.to(device)
                        _logit = model(x, max_len)
                        _loss = test_criterion(_logit, _y)
                        pred = np.argmax(_logit.cpu().numpy(), axis=1)
                        truth = _y.cpu().numpy()
                        running_f1 += f1_score(truth, pred, average='weighted')
                        running_acc += accuracy_score(truth, pred)
                        running_rec += recall_score(truth, pred, average="weighted", zero_division=0)
                        running_pre += precision_score(truth, pred, average="weighted", zero_division=0)
                        valid_running_loss += _loss.mean(0).item()
                model.train()

                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(test_loader) 
                average_f1 = running_f1 / len(test_loader)
                average_acc = running_acc / len(test_loader)
                average_rec = running_rec / len(test_loader)
                average_pre = running_pre / len(test_loader)
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, F1: {:.4f}, Acc: {:.4f}, Recall: {:.4f}, Precision: {:.4f}'
                            .format(epoch+1, n_epochs, global_step, n_epochs*len(data_loader),
                                    average_train_loss, average_valid_loss, average_f1, average_acc, average_rec, average_pre))