import math
import torch
from sentence_transformers import models, losses, SentenceTransformer, CrossEncoder
from dataset import BI_Encoder
from torch.utils.data import DataLoader


use_cuda = torch.cuda.is_available()
max_seq_length = 256
num_epochs = 10
batch_size = 8
model_name = 'bert-base-uncased'

cross_encoder_path = f'./output/cross_{model_name}/'
bi_encoder_path = f'./output/bi_{model_name}/'

crossencoder = CrossEncoder(cross_encoder_path)
dataset = BI_Encoder()
dataset.label(crossencoder)
train_dataloader = DataLoader(dataset, batch_size=batch_size)


word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length, do_lower_case=True)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_loss = losses.MultipleNegativesRankingLoss(bi_encoder)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)], warmup_steps=warmup_steps, epochs=num_epochs, output_path=bi_encoder_path)
