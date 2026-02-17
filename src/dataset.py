
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Map string labels to integers if needed
        # Assuming Data Ingestion gives 'Positive', 'Negative', 'Neutral'
        self.label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        # Handle label encoding
        if isinstance(label, str):
            label = self.label_map.get(label, 1) # Default to Neutral if unknown
        elif isinstance(label, (int, float)):
            # Map 1-5 ratings to 0-2 (Negative, Neutral, Positive)
            if label <= 2:
                label = 0
            elif label == 3:
                label = 1
            else:
                label = 2
            
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
