import torch
import numpy as np
import evaluate
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import pipeline, Trainer
from transformers import AutoConfig
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import os

#修改ModifiedModel类中的hook位置，观察每一层输入输出

os.environ["TOKENIZERS_PARALLELISM"] = "true"
'''--------------------model load----------------------'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './distilbert-base-multilingual-cased-sentiments-student/'
config = AutoConfig.from_pretrained("./distilbert-base-multilingual-cased-sentiments-student/config.json")
tokenizer_path = './distilbert-base-multilingual-cased-sentiments-student/'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

from transformers.modeling_outputs import SequenceClassifierOutput


class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.transformer_query_input = None
        self.transformer_query_output = None
        self.original_model = original_model
        self.original_model.distilbert.transformer.layer[5].attention.q_lin.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.layer_input = input
        self.layer_output = output
    # def transformer_hook(self, module, input, output):
    #     self.transformer_output = output

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                      labels=labels)
        hidden_states = outputs.hidden_states

        transformer6_output = hidden_states[6]
        print("transformer6_output:", transformer6_output)
        print("transformer6_output_shape:", transformer6_output.shape)
        # 获取transformer.attention.q_lin的输入输出
        if isinstance(self.layer_input, tuple):
            input = self.layer_input[0]
        else:
            input=self.layer_input
        print("input.shape:",input.shape)
        if isinstance(self.layer_output, tuple):
            output = self.layer_output[0]
        else:
            output = self.layer_output

        print("output.shape:",output.shape)
        print("output:",output)




        # 如果提供了labels，计算loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = outputs.logits
            loss = loss_fct(logits.view(-1, self.original_model.config.num_labels), labels.view(-1))
            return SequenceClassifierOutput(
                loss=loss,
                logits=outputs,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(
                logits=outputs,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


model = ModifiedModel(model)
model.to(device)
print(model)
'''--------------------load data----------------------'''
# import pyarrow.parquet as pq
#
train_data = pd.read_excel('./Datasets/glue/train_data.xlsx')
# train_file = pq.ParquetFile('./Datasets/glue/train-00000-of-00001.parquet')
# train_data = train_file.read().to_pandas()
#
test_data = pd.read_excel('./Datasets/glue/validation_data.xlsx')


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

# for name, param in model.named_parameters():
#     param.requires_grad = False
#     if 'fc' in name:
#         param.requires_grad = True
# # # 打印出所有需要训练的参数
# for name, param in model.named_parameters():
#     if param.requires_grad == True:
#         print(name)


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


training_args = TrainingArguments(
    output_dir="./output/freeze_trainer",
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    save_strategy="steps",  # 设置为"steps"
    save_steps=1e10,  # 设置为一个非常大的数
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5)
trainer = Trainer(

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

'''--------------------eval model----------------------'''

# 创建一个DataLoader
dataloader = DataLoader(test_dataset, batch_size=64)

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
