import math
import torch
from sentence_transformers import models, losses, SentenceTransformer
from dataset import BI_Encoder
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

root = '.'
use_cuda = torch.cuda.is_available()
num_epochs = 10
# batch_size = 128
batch_size = 16
cross_model_name = 'cross-encoder/stsb-distilroberta-base'
cross_encoder_path = f'{root}/output/cross_{cross_model_name}/'
# bi_model_name = 'nli-distilroberta-base-v2'
bi_model_name = 'all-distilroberta-v1'
bi_encoder_path = f'{root}/output/bi_{bi_model_name}/'


dataset = BI_Encoder(supervised_dataset=True)
ts1, ts2, tScore = dataset.test()
evaluator = EmbeddingSimilarityEvaluator(ts1, ts2, tScore, batch_size=2, show_progress_bar=True)
train_dataloader = DataLoader(dataset, batch_size=batch_size)
bi_encoder = SentenceTransformer(bi_model_name)
bi_encoder.max_seq_length = 256
train_loss = losses.MultipleNegativesRankingLoss(bi_encoder)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)],
                    evaluator=evaluator,
                    evaluation_steps=5000,
                    warmup_steps=warmup_steps, 
                    epochs=num_epochs, 
                    output_path=bi_encoder_path, 
                    use_amp=True, 
                    show_progress_bar=True,
            )

bi_encoder.save(f'{root}/output/bi_out')
print('model done')