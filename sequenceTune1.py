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

# 每一epoch从transformer内部获取数据，最终求和输入到pre_classifier同时训练

os.environ["TOKENIZERS_PARALLELISM"] = "true"
'''--------------------model load----------------------'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './distilbert-base-multilingual-cased-sentiments-student/'
config = AutoConfig.from_pretrained("./distilbert-base-multilingual-cased-sentiments-student/config.json")
tokenizer_path = './distilbert-base-multilingual-cased-sentiments-student/'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

from transformers.modeling_outputs import SequenceClassifierOutput


class ModifiedModel1(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel1, self).__init__()
        self.original_model = original_model
        self.pre_classifier = original_model.pre_classifier
        self.classifier=original_model.classifier
        self.dropout=original_model.dropout
    def forward(self, input_ids, attention_mask, labels=None,output_hidden_states=True):
        outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                      labels=labels)
        hidden_states = outputs.hidden_states
        transformer1_output = hidden_states[1]
        transformer1_output = transformer1_output[:, 0, :]

        pre_classifier_input = transformer1_output
        pre_classifier_output = self.original_model.pre_classifier(pre_classifier_input)
        # 将全连接层的输出和pre_classifier的输出一起输入到classifier中
        # classifier_input = pre_classifier_output
        # classifier_input = fc_4_output +fc_5_output+fc_3_output+fc_2_output+ pre_classifier_output
        # print("classifier_input_shape:", classifier_input.shape)
        classifier_output = self.original_model.classifier(pre_classifier_output)
        # print("classifier_output_shape:", classifier_output.shape)
        dropout_output = self.original_model.dropout(classifier_output)
        # print("logits_shape:", dropout_output.shape)

        # 如果提供了labels，计算loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(dropout_output.view(-1, self.original_model.config.num_labels), labels.view(-1))
            return SequenceClassifierOutput(
                loss=loss,
                logits=dropout_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(
                logits=dropout_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

model = ModifiedModel1(model)
# print(model)
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
    logits = logits[0].tolist()  # 使用logits元组的第一个元素
    predictions = np.argmax(logits, axis=-1)
    # print(predictions, labels)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./output/temp_trainer",
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    save_strategy="steps",  # 设置为"steps"
    save_steps=1e10,  # 设置为一个非常大的数
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2)


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
    if 'layer.0.attention' in name:
        param.requires_grad = True
# # 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')
'''--------------------model eval----------------------'''
# print("evaluating...")
# dataloader = DataLoader(test_dataset, batch_size=64)
# # 初始化一个列表来保存所有的预测结果
# all_predictions = []
# all_labels = []
# # 将模型设置为评估模式
# model.eval()
# # 遍历测试集
# for batch in dataloader:
#     # 将数据移动到设备上
#     inputs = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     labels = batch['labels'].to(device)
#     # 不需要计算梯度
#     with torch.no_grad():
#         # 前向传播
#         outputs = model(inputs, attention_mask=attention_mask)
#     # 获取预测结果
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     # 将预测结果和真实标签保存起来
#     all_predictions.extend(predictions.cpu().numpy())
#     all_labels.extend(labels.cpu().numpy())
# from sklearn.metrics import accuracy_score
# # 计算准确度
# accuracy = accuracy_score(all_labels, all_predictions)
# print("Accuracy:", accuracy)
del model
'''--------------------epoch2----------------------'''

class ModifiedModel2(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel2, self).__init__()
        self.original_model = original_model
        self.pre_classifier = original_model.pre_classifier
        self.classifier=original_model.classifier
        self.dropout=original_model.dropout
    def forward(self, input_ids, attention_mask, labels=None,output_hidden_states=True):
        outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                      labels=labels)
        hidden_states = outputs.hidden_states
        transformer2_output = hidden_states[2]
        transformer2_output = transformer2_output[:, 0, :]

        pre_classifier_input = transformer2_output
        pre_classifier_output = self.original_model.pre_classifier(pre_classifier_input)
        # 将全连接层的输出和pre_classifier的输出一起输入到classifier中
        # classifier_input = pre_classifier_output
        # classifier_input = fc_4_output +fc_5_output+fc_3_output+fc_2_output+ pre_classifier_output
        # print("classifier_input_shape:", classifier_input.shape)
        classifier_output = self.original_model.classifier(pre_classifier_output)
        # print("classifier_output_shape:", classifier_output.shape)
        dropout_output = self.original_model.dropout(classifier_output)
        # print("logits_shape:", dropout_output.shape)

        # 如果提供了labels，计算loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(dropout_output.view(-1, self.original_model.config.num_labels), labels.view(-1))
            return SequenceClassifierOutput(
                loss=loss,
                logits=dropout_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(
                logits=dropout_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
model = AutoModelForSequenceClassification.from_pretrained(model_path)
original_state_dict = torch.load('./output/sequenceTune/model.pth')
# 创建一个新的字典来保存修改后的参数
new_state_dict = OrderedDict()
# 遍历原始的模型参数
for param_name, param_value in original_state_dict.items():
    # 删除参数名称中的'base_model.model.'前缀
    new_param_name = param_name.replace('base_model.model.', '')
    new_param_name = new_param_name.replace('original_model.', '')
    # 将修改后的参数添加到新的字典中
    new_state_dict[new_param_name] = param_value
model.load_state_dict(new_state_dict)
model = ModifiedModel2(model)

training_args = TrainingArguments(
    output_dir="./output/temp_trainer",
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    save_strategy="steps",  # 设置为"steps"
    save_steps=1e10,  # 设置为一个非常大的数
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2)


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
    if 'layer.1.attention' in name:
        param.requires_grad = True
# # 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')
'''--------------------model eval----------------------'''
# print("evaluating...")
# dataloader = DataLoader(test_dataset, batch_size=64)
# # 初始化一个列表来保存所有的预测结果
# all_predictions = []
# all_labels = []
# # 将模型设置为评估模式
# model.eval()
# # 遍历测试集
# for batch in dataloader:
#     # 将数据移动到设备上
#     inputs = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     labels = batch['labels'].to(device)
#     # 不需要计算梯度
#     with torch.no_grad():
#         # 前向传播
#         outputs = model(inputs, attention_mask=attention_mask)
#     # 获取预测结果
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     # 将预测结果和真实标签保存起来
#     all_predictions.extend(predictions.cpu().numpy())
#     all_labels.extend(labels.cpu().numpy())
# from sklearn.metrics import accuracy_score
# # 计算准确度
# accuracy = accuracy_score(all_labels, all_predictions)
# print("Accuracy:", accuracy)
del model
'''--------------------epoch3----------------------'''

class ModifiedModel3(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel3, self).__init__()
        self.original_model = original_model
        self.pre_classifier = original_model.pre_classifier
        self.classifier=original_model.classifier
        self.dropout=original_model.dropout
    def forward(self, input_ids, attention_mask, labels=None,output_hidden_states=True):
        outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                      labels=labels)
        hidden_states = outputs.hidden_states
        transformer3_output = hidden_states[3]
        transformer3_output = transformer3_output[:, 0, :]

        pre_classifier_input = transformer3_output
        pre_classifier_output = self.original_model.pre_classifier(pre_classifier_input)
        # 将全连接层的输出和pre_classifier的输出一起输入到classifier中
        # classifier_input = pre_classifier_output
        # classifier_input = fc_4_output +fc_5_output+fc_3_output+fc_2_output+ pre_classifier_output
        # print("classifier_input_shape:", classifier_input.shape)
        classifier_output = self.original_model.classifier(pre_classifier_output)
        # print("classifier_output_shape:", classifier_output.shape)
        dropout_output = self.original_model.dropout(classifier_output)
        # print("logits_shape:", dropout_output.shape)

        # 如果提供了labels，计算loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(dropout_output.view(-1, self.original_model.config.num_labels), labels.view(-1))
            return SequenceClassifierOutput(
                loss=loss,
                logits=dropout_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(
                logits=dropout_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


model = AutoModelForSequenceClassification.from_pretrained(model_path)
original_state_dict = torch.load('./output/sequenceTune/model.pth')
# 创建一个新的字典来保存修改后的参数
new_state_dict = OrderedDict()
# 遍历原始的模型参数
for param_name, param_value in original_state_dict.items():
    # 删除参数名称中的'base_model.model.'前缀
    new_param_name = param_name.replace('base_model.model.', '')
    new_param_name = new_param_name.replace('original_model.', '')
    # 将修改后的参数添加到新的字典中
    new_state_dict[new_param_name] = param_value
model.load_state_dict(new_state_dict)
model = ModifiedModel3(model)

training_args = TrainingArguments(
    output_dir="./output/temp_trainer",
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    save_strategy="steps",  # 设置为"steps"
    save_steps=1e10,  # 设置为一个非常大的数
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2)


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
    if 'layer.2.attention' in name:
        param.requires_grad = True
# # 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')
'''--------------------model eval----------------------'''
# print("evaluating...")
# dataloader = DataLoader(test_dataset, batch_size=64)
# # 初始化一个列表来保存所有的预测结果
# all_predictions = []
# all_labels = []
# # 将模型设置为评估模式
# model.eval()
# # 遍历测试集
# for batch in dataloader:
#     # 将数据移动到设备上
#     inputs = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     labels = batch['labels'].to(device)
#     # 不需要计算梯度
#     with torch.no_grad():
#         # 前向传播
#         outputs = model(inputs, attention_mask=attention_mask)
#     # 获取预测结果
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     # 将预测结果和真实标签保存起来
#     all_predictions.extend(predictions.cpu().numpy())
#     all_labels.extend(labels.cpu().numpy())
# from sklearn.metrics import accuracy_score
# # 计算准确度
# accuracy = accuracy_score(all_labels, all_predictions)
# print("Accuracy:", accuracy)
del model
'''--------------------epoch4----------------------'''

class ModifiedModel4(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel4, self).__init__()
        self.original_model = original_model
        self.pre_classifier = original_model.pre_classifier
        self.classifier=original_model.classifier
        self.dropout=original_model.dropout
    def forward(self, input_ids, attention_mask, labels=None,output_hidden_states=True):
        outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                      labels=labels)
        hidden_states = outputs.hidden_states
        transformer4_output = hidden_states[4]
        transformer4_output = transformer4_output[:, 0, :]

        pre_classifier_input = transformer4_output
        pre_classifier_output = self.original_model.pre_classifier(pre_classifier_input)
        # 将全连接层的输出和pre_classifier的输出一起输入到classifier中
        # classifier_input = pre_classifier_output
        # classifier_input = fc_4_output +fc_5_output+fc_3_output+fc_2_output+ pre_classifier_output
        # print("classifier_input_shape:", classifier_input.shape)
        classifier_output = self.original_model.classifier(pre_classifier_output)
        # print("classifier_output_shape:", classifier_output.shape)
        dropout_output = self.original_model.dropout(classifier_output)
        # print("logits_shape:", dropout_output.shape)

        # 如果提供了labels，计算loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(dropout_output.view(-1, self.original_model.config.num_labels), labels.view(-1))
            return SequenceClassifierOutput(
                loss=loss,
                logits=dropout_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(
                logits=dropout_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


model = AutoModelForSequenceClassification.from_pretrained(model_path)
original_state_dict = torch.load('./output/sequenceTune/model.pth')
# 创建一个新的字典来保存修改后的参数
new_state_dict = OrderedDict()
# 遍历原始的模型参数
for param_name, param_value in original_state_dict.items():
    # 删除参数名称中的'base_model.model.'前缀
    new_param_name = param_name.replace('base_model.model.', '')
    new_param_name = new_param_name.replace('original_model.', '')
    # 将修改后的参数添加到新的字典中
    new_state_dict[new_param_name] = param_value
model.load_state_dict(new_state_dict)
model = ModifiedModel4(model)


training_args = TrainingArguments(
    output_dir="./output/temp_trainer",
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    save_strategy="steps",  # 设置为"steps"
    save_steps=1e10,  # 设置为一个非常大的数
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5)


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
    if 'layer.3.attention' in name:
        param.requires_grad = True
# # 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')
'''--------------------model eval----------------------'''
# print("evaluating...")
# dataloader = DataLoader(test_dataset, batch_size=64)
# # 初始化一个列表来保存所有的预测结果
# all_predictions = []
# all_labels = []
# # 将模型设置为评估模式
# model.eval()
# # 遍历测试集
# for batch in dataloader:
#     # 将数据移动到设备上
#     inputs = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     labels = batch['labels'].to(device)
#     # 不需要计算梯度
#     with torch.no_grad():
#         # 前向传播
#         outputs = model(inputs, attention_mask=attention_mask)
#     # 获取预测结果
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     # 将预测结果和真实标签保存起来
#     all_predictions.extend(predictions.cpu().numpy())
#     all_labels.extend(labels.cpu().numpy())
# from sklearn.metrics import accuracy_score
# # 计算准确度
# accuracy = accuracy_score(all_labels, all_predictions)
# print("Accuracy:", accuracy)
del model
'''--------------------epoch5----------------------'''

class ModifiedModel5(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel5, self).__init__()
        self.original_model = original_model
        self.pre_classifier = original_model.pre_classifier
        self.classifier=original_model.classifier
        self.dropout=original_model.dropout
    def forward(self, input_ids, attention_mask, labels=None,output_hidden_states=True):
        outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                      labels=labels)
        hidden_states = outputs.hidden_states
        transformer5_output = hidden_states[5]
        transformer5_output = transformer5_output[:, 0, :]

        pre_classifier_input = transformer5_output
        pre_classifier_output = self.original_model.pre_classifier(pre_classifier_input)
        # 将全连接层的输出和pre_classifier的输出一起输入到classifier中
        # classifier_input = pre_classifier_output
        # classifier_input = fc_4_output +fc_5_output+fc_3_output+fc_2_output+ pre_classifier_output
        # print("classifier_input_shape:", classifier_input.shape)
        classifier_output = self.original_model.classifier(pre_classifier_output)
        # print("classifier_output_shape:", classifier_output.shape)
        dropout_output = self.original_model.dropout(classifier_output)
        # print("logits_shape:", dropout_output.shape)

        # 如果提供了labels，计算loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(dropout_output.view(-1, self.original_model.config.num_labels), labels.view(-1))
            return SequenceClassifierOutput(
                loss=loss,
                logits=dropout_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(
                logits=dropout_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

model = AutoModelForSequenceClassification.from_pretrained(model_path)
original_state_dict = torch.load('./output/sequenceTune/model.pth')
# 创建一个新的字典来保存修改后的参数
new_state_dict = OrderedDict()
# 遍历原始的模型参数
for param_name, param_value in original_state_dict.items():
    # 删除参数名称中的'base_model.model.'前缀
    new_param_name = param_name.replace('base_model.model.', '')
    new_param_name = new_param_name.replace('original_model.', '')
    # 将修改后的参数添加到新的字典中
    new_state_dict[new_param_name] = param_value
model.load_state_dict(new_state_dict)
model = ModifiedModel5(model)

training_args = TrainingArguments(
    output_dir="./output/temp_trainer",
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    save_strategy="steps",  # 设置为"steps"
    save_steps=1e10,  # 设置为一个非常大的数
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5)


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
    if 'layer.4.attention' in name:
        param.requires_grad = True
# 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')
'''--------------------model eval----------------------'''
# print("evaluating...")
# dataloader = DataLoader(test_dataset, batch_size=64)
# # 初始化一个列表来保存所有的预测结果
# all_predictions = []
# all_labels = []
# # 将模型设置为评估模式
# model.eval()
# # 遍历测试集
# for batch in dataloader:
#     # 将数据移动到设备上
#     inputs = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     labels = batch['labels'].to(device)
#     # 不需要计算梯度
#     with torch.no_grad():
#         # 前向传播
#         outputs = model(inputs, attention_mask=attention_mask)
#     # 获取预测结果
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     # 将预测结果和真实标签保存起来
#     all_predictions.extend(predictions.cpu().numpy())
#     all_labels.extend(labels.cpu().numpy())
# from sklearn.metrics import accuracy_score
# # 计算准确度
# accuracy = accuracy_score(all_labels, all_predictions)
# print("Accuracy:", accuracy)
del model
'''-----------------------------------epoch6--------------------------------------------'''
class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.pre_classifier = original_model.pre_classifier
        self.classifier=original_model.classifier
        self.dropout=original_model.dropout
    def forward(self, input_ids, attention_mask, labels=None,output_hidden_states=True):
        outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                      labels=labels)
        # # 如果提供了labels，计算loss
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(dropout_output.view(-1, self.original_model.config.num_labels), labels.view(-1))
        #     return SequenceClassifierOutput(
        #         loss=loss,
        #         logits=dropout_output,
        #         hidden_states=outputs.hidden_states,
        #         attentions=outputs.attentions,
        #     )
        # else:
        return SequenceClassifierOutput(
            logits=outputs.logits,
            loss=outputs.loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model = AutoModelForSequenceClassification.from_pretrained(model_path)
original_state_dict = torch.load('./output/sequenceTune/model.pth')
# 创建一个新的字典来保存修改后的参数
new_state_dict = OrderedDict()
# 遍历原始的模型参数
for param_name, param_value in original_state_dict.items():
    # 删除参数名称中的'base_model.model.'前缀
    new_param_name = param_name.replace('base_model.model.', '')
    new_param_name = new_param_name.replace('original_model.', '')
    # 将修改后的参数添加到新的字典中
    new_state_dict[new_param_name] = param_value
model.load_state_dict(new_state_dict)
model = ModifiedModel(model)
training_args = TrainingArguments(
    output_dir="./output/temp_trainer",
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    save_strategy="steps",  # 设置为"steps"
    save_steps=1e10,  # 设置为一个非常大的数
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5)
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
    if 'layer.5.attention' in name or 'classifier' in name or 'dropout' in name:
        param.requires_grad = True
# 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')

'''--------------------model eval----------------------'''
# print("evaluating...")
# dataloader = DataLoader(test_dataset, batch_size=64)
# # 初始化一个列表来保存所有的预测结果
# all_predictions = []
# all_labels = []
# # 将模型设置为评估模式
# model.eval()
# # 遍历测试集
# for batch in dataloader:
#     # 将数据移动到设备上
#     inputs = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     labels = batch['labels'].to(device)
#     # 不需要计算梯度
#     with torch.no_grad():
#         # 前向传播
#         outputs = model(inputs, attention_mask=attention_mask)
#     # 获取预测结果
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     # 将预测结果和真实标签保存起来
#     all_predictions.extend(predictions.cpu().numpy())
#     all_labels.extend(labels.cpu().numpy())
# from sklearn.metrics import accuracy_score
# # 计算准确度
# accuracy = accuracy_score(all_labels, all_predictions)
# print("Accuracy:", accuracy)

