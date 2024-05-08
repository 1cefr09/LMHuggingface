import torch
import numpy as np
import evaluate
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset,DataLoader
from transformers import pipeline, Trainer
from transformers import AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


#只推理

'''--------------------model load----------------------'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './distilbert-base-multilingual-cased-sentiments-student/'
# model_path = './output/freeze_trainer/checkpoint-2106/'
config = AutoConfig.from_pretrained("./distilbert-base-multilingual-cased-sentiments-student/config.json")
tokenizer_path = './distilbert-base-multilingual-cased-sentiments-student/'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model.to(device)
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


metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits[0].tolist()  # 使用logits元组的第一个元素
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 创建一个DataLoader
dataloader = DataLoader(test_dataset, batch_size=64)

# 初始化一个列表来保存所有的预测结果和真实标签
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
        print(outputs)
    # 获取预测结果
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # 将预测结果和真实标签保存起来
    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# 计算准确度
accuracy = accuracy_score(all_labels, all_predictions)
print("Accuracy:", accuracy)
# # 从训练数据中取出
# batch_data = train_data.iloc[:64]
#
# # 创建数据集
# batch_dataset = DataFrameDataset(batch_data, tokenizer)
#
# # 创建 DataLoader
# from torch.utils.data import DataLoader
#
# dataloader = DataLoader(batch_dataset, batch_size=64)
#
# # 获取一批数据
# batch = next(iter(dataloader))
#
# # 将数据输入模型
# inputs = batch['input_ids'].to(device)
# attention_mask = batch['attention_mask'].to(device)
# labels = batch['labels'].to(device)
#
# with torch.no_grad():
#     model(inputs, attention_mask=attention_mask, labels=labels)
#
# # 获取模型的输出
# output = model(inputs, attention_mask=attention_mask, labels=labels)
# print('output', output)
# print(output.logits.shape)
# out=output.logits.cpu().detach().numpy()
# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)
#
# softmax_out = np.apply_along_axis(softmax, 1, out)
# true_labels = batch['labels'].numpy()
#
# # 使用 argmax 找到预测的类别
# pred_labels = np.argmax(softmax_out, axis=1)
#
# # 计算预测正确的样本数量
# correct_predictions = np.sum(pred_labels == true_labels)
#
# # 计算准确率
# accuracy = correct_predictions / len(true_labels)
#
# print("Accuracy:", accuracy)