from transformers import BertTokenizer, BertModel
from src.config import device

def get_text_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    return tokenizer, model
