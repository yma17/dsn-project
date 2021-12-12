import math
import torch
from sentence_transformers import models, losses, SentenceTransformer
from dataset import BI_Encoder
from torch.utils.data import DataLoader


use_cuda = torch.cuda.is_available()
max_seq_length = 256
num_epochs = 10
batch_size = 8
#model_name = 'bert-base-uncased'
cross_model_name = 'cross-encoder/stsb-distilroberta-base'
cross_encoder_path = f'./output/cross_{cross_model_name}/'
bi_model_name = 'all-distilroberta-v1'
bi_encoder_path = f'./output/bi_{bi_model_name}/'


dataset = BI_Encoder()
train_dataloader = DataLoader(dataset, batch_size=batch_size)

word_embedding_model = models.Transformer(bi_model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_loss = losses.MultipleNegativesRankingLoss(bi_encoder)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)], warmup_steps=warmup_steps, epochs=num_epochs, output_path=bi_encoder_path)
