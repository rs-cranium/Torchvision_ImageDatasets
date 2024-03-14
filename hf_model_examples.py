### Method 1: Pipelines ###

from transformers import pipeline

# Example: Text generation
text_generator = pipeline("text-generation", model="gpt2")
generated_text = text_generator("Once upon a time")
print(generated_text)

### Method 2: Model Classes ###

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Example: GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

### Method 3: Auto Models ###

from transformers import AutoModelForSequenceClassification

# Example: AutoModel
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

### Method 4: Trainer ###

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

# Example: Fine-tuning BERT
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

### Method 5: Model Hub ###

from transformers import AutoModelForSequenceClassification

# Example: Load model from Model Hub
model_identifier = "textattack/bert-base-uncased-ag-news"
model = AutoModelForSequenceClassification.from_pretrained(model_identifier)

### Method 6: Custom Models ###

from transformers import BertModel, BertTokenizer
import torch

# Example: Custom model using BERT base
class CustomBERTModel(torch.nn.Module):
    def __init__(self):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Take the pooled output
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits

# Usage
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = CustomBERTModel()
input_ids = tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"]
attention_mask = tokenizer("Hello, how are you?", return_tensors="pt")["attention_mask"]
outputs = model(input_ids, attention_mask)
print(outputs)
