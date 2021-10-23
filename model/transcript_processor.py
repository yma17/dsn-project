from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("ghanashyamvtatti/roberta-fake-news")
fake_news_model = AutoModelForSequenceClassification.from_pretrained("ghanashyamvtatti/roberta-fake-news")


