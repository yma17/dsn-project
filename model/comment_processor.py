from transformers import AutoTokenizer, AutoModelForSequenceClassification  
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-emotion-analysis")
emotion_model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-emotion-analysis")
