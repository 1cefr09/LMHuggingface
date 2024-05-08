import torch
import numpy as np
import evaluate
import pandas as pd

from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import Dataset
from transformers import pipeline, Trainer
from transformers import AutoConfig
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import os


os.environ["TOKENIZERS_PARALLELISM"] = "true"
'''--------------------model load----------------------'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './distilbert-base-multilingual-cased-sentiments-student/'
config = AutoConfig.from_pretrained("./distilbert-base-multilingual-cased-sentiments-student/config.json")
tokenizer_path = './distilbert-base-multilingual-cased-sentiments-student/'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

'''--------------------load data----------------------'''
# import pyarrow.parquet as pq
#
train_data = pd.read_excel('./Datasets/glue/train_data.xlsx')
# train_data = Dataset.from_pandas(train_data)
# train_file = pq.ParquetFile('./Datasets/glue/train-00000-of-00001.parquet')
# train_data = train_file.read().to_pandas()
#
test_data = pd.read_excel('./Datasets/glue/validation_data.xlsx')
# test_data = Dataset.from_pandas(test_data)
from torch.utils.data import Dataset

class DataFrameDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(row['sentence'], padding="max_length", truncation=True)
        item = {key: torch.tensor(val) for key, val in inputs.items()}
        item['labels'] = torch.tensor(row['label'])
        return item

    def __len__(self):
        return len(self.df)

train_dataset = DataFrameDataset(train_data, tokenizer)
test_dataset = DataFrameDataset(test_data, tokenizer)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.tolist()  # 使用logits元组的第一个元素
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./output/temp_trainer",
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    save_strategy="steps",  # 设置为"steps"
    save_steps=1e10,  # 设置为一个非常大的数
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3)


'''--------------------train----------------------'''
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)
model.to(device)
print("device:", device)
print(model)

for name, param in model.named_parameters():
    param.requires_grad = False
    if 'layer.5' in name or 'classifier' in name or 'dropout' in name:
        param.requires_grad = True
# # 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')

'''--------------------train----------------------'''

for name, param in model.named_parameters():
    param.requires_grad = False
    if 'layer.4' in name:
        param.requires_grad = True
# # 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')


'''--------------------train----------------------'''

for name, param in model.named_parameters():
    param.requires_grad = False
    if 'layer.3' in name:
        param.requires_grad = True
# # 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')

'''--------------------train----------------------'''
for name, param in model.named_parameters():
    param.requires_grad = False
    if 'layer.2' in name:
        param.requires_grad = True
# # 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')


'''--------------------train----------------------'''
for name, param in model.named_parameters():
    param.requires_grad = False
    if 'layer.1' in name:
        param.requires_grad = True
# 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')


'''--------------------train----------------------'''
for name, param in model.named_parameters():
    param.requires_grad = False
    if 'layer.0' in name:
        param.requires_grad = True
# 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')

