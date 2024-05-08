import torch
import numpy as np
import evaluate
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline, Trainer
from transformers import AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import os

#正常方法训练、全参或冻结


os.environ["TOKENIZERS_PARALLELISM"] = "true"
'''--------------------model load----------------------'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './distilbert-base-multilingual-cased-sentiments-student/'
# config = AutoConfig.from_pretrained("./distilbert-base-multilingual-cased-sentiments-student/config.json")
tokenizer_path = './distilbert-base-multilingual-cased-sentiments-student/'
# model = DistilBertForMaskedLM.from_from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained("./distilbert-base-multilingual-cased-sentiments-student/")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


model.add_adapter("my_adapter", AdapterType.text_task)
print(model)

'''--------------------pipeline----------------------'''
# sentiment_analysis = pipeline('sentiment-analysis', model=model_path, tokenizer=tokenizer_path)
# print(sentiment_analysis(' hide new secretions from the parental units'))
# print(tokenizer(' anguish , anger and frustration'))

'''--------------------load data----------------------'''
import pyarrow.parquet as pq

train_data = pd.read_excel('./Datasets/glue/train_data.xlsx')
# train_file = pq.ParquetFile('./Datasets/glue/train-00000-of-00001.parquet')
# train_data = train_file.read().to_pandas()

test_data = pd.read_excel('./Datasets/glue/validation_data.xlsx')


# test_file = pq.ParquetFile('./Datasets/glue/validation-00000-of-00001.parquet')
# test_data = test_file.read().to_pandas()
# train_data=pd.read_excel('./Datasets/glue/train_data.xlsx')
# test_data = pd.read_excel('./Datasets/glue/validation_data.xlsx')


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


'''--------------------training args----------------------'''

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print("logits:", logits)
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model.train_adapter("my_adapter")
# 冻结
# for name, param in model.named_parameters():
#     # 检查层的名称，如果层的名称包含 'layer.0.' 或 'layer.1.'，则冻结该层
#     if 'layer.0.' in name or 'layer.1.' in name or 'layer.2.' in name or 'layer.3.' in name or 'layer.4.' in name:
#         param.requires_grad = False
#     else:
#         param.requires_grad = True
#
# 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)

training_args = TrainingArguments(
    output_dir="./output/test_trainer",
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    save_strategy="steps",  # 设置为"steps"
    save_steps=1e10,  # 设置为一个非常大的数
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=2)
trainer = Trainer(
    with_adapters=True,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)
model.to(device)
print("device:", device)

'''--------------------train----------------------'''
print("training...")
trainer.train()
model.save_pretrained("./output/adapter_trainer/model")


dataloader = DataLoader(test_dataset, batch_size=32)

# 初始化一个列表来保存所有的预测结果
all_predictions = []
all_labels = []

# 将模型设置为评估模式
model.eval()

# 遍历测试集
for batch in dataloader:
    # 将数据移动到设备上
    inputs = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # 不需要计算梯度
    with torch.no_grad():
        # 前向传播
        outputs = model(inputs, attention_mask=attention_mask)

    # 获取预测结果
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # 将预测结果和真实标签保存起来
    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())
from sklearn.metrics import accuracy_score
# 计算准确度
accuracy = accuracy_score(all_labels, all_predictions)
print("Accuracy:", accuracy)
