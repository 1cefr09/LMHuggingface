import torch
import numpy as np
import evaluate
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import pipeline, Trainer
from transformers import AutoConfig
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

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
# train_file = pq.ParquetFile('./Datasets/glue/train-00000-of-00001.parquet')
# train_data = train_file.read().to_pandas()
#
test_data = pd.read_excel('./Datasets/glue/validation_data.xlsx')
'''--------------------modify model----------------------'''

class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.fc_4 = nn.Linear(768, 768)  # 假设transformer.layer.4的输出维度为768
        self.fc_3 = nn.Linear(768, 768)  # 假设transformer.layer.3的输出维度为768
        self.original_model.pre_classifier.register_forward_hook(self.pre_classifier_hook)

    def pre_classifier_hook(self, module, input, output):
            self.pre_classifier_output = output

    def forward(self, input_ids, attention_mask):
        outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        # 获取transformer.layer.4的输出
        intermediate_output4 = hidden_states[4]
        intermediate_output3 = hidden_states[3]
        # 将transformer.layer.4的输出作为全连接层的输入
        fc_4_output = self.fc_4(intermediate_output4)
        fc_3_output = self.fc_3(intermediate_output3)
        # 获取pre_classifier的输出
        pre_classifier_output = self.pre_classifier_output
        # 将全连接层的输出和pre_classifier的输出一起输入到classifier中
        classifier_input = torch.cat((fc_4_output, fc_3_output), dim=-1)
        classifier_input = torch.cat((classifier_input, pre_classifier_output), dim=-1)
        classifier_output = self.original_model.classifier(classifier_input)

        return classifier_output
# classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(768, 768)),
#                           ('relu1', nn.ReLU()),
#                           ('dropout1',nn.Dropout(0.1)),
#                           ('fc2', nn.Linear(768, 768)),
#                             ('relu2', nn.ReLU()),
#                           ('dropout2',nn.Dropout(0.1)),
#                             ('fc3', nn.Linear(768, 768)),
#                              ('relu3', nn.ReLU()),
#                             ('dropout3', nn.Dropout(0.1)),
#                             ('fc4', nn.Linear(768, 768)),
#                             ('relu4', nn.ReLU()),
#                           ('dropout4',nn.Dropout(0.1)),
#                             ('fc5', nn.Linear(768, 768)),
#                             ('relu5', nn.ReLU()),
#                           ('dropout5',nn.Dropout(0.1)),
#                             ('fc6', nn.Linear(768, 768)),
#                             ('relu6', nn.ReLU()),
#                           ('dropout6',nn.Dropout(0.1)),
#                             ('fc7', nn.Linear(768, 768)),
#                             ('relu7', nn.ReLU()),
#                           ('dropout7',nn.Dropout(0.1)),
#                             ('fc8', nn.Linear(768, 768)),
#                              ('relu8', nn.ReLU()),
#                             ('dropout8', nn.Dropout(0.1)),
#                             ('fc9', nn.Linear(768, 768)),
#                             ('relu9', nn.ReLU()),
#                             ('dropout9', nn.Dropout(0.1)),
#                              ('fc10', nn.Linear(768, 3))
#                           ]))
# model.classifier = classifier
model=ModifiedModel(model)
print(model)

for name, param in model.named_parameters():
    param.requires_grad = False
    if 'fc' in name:
        param.requires_grad = True
# 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)

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
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
                  output_dir="./output/freeze_trainer",
                  evaluation_strategy="epoch",
                  # save_strategy="epoch",
                  save_strategy="steps",  # 设置为"steps"
                  save_steps=1e10,  # 设置为一个非常大的数
                  per_device_train_batch_size=64,
                  per_device_eval_batch_size=64)
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
model.save_pretrained('./output/freeze_trainer/model')
torch.save(model, './output/freeze_trainer/model.pth')


# model = torch.load('./output/freeze_trainer/model.pth')
# model.to(device)
# print(model)
# class ForwardHook():
#     def __init__(self, module):
#         self.hook = module.register_forward_hook(self.hook_func)
#
#     def hook_func(self, module, input, output):
#         self.output = output
#
#     def close(self):
#         self.hook.remove()
#
#
# # 创建钩子
# hook = ForwardHook(model.classifier.fc9)
#
# # 从训练数据中取出
# batch_data = test_data.iloc[:600]
#
# # 创建数据集
# batch_dataset = DataFrameDataset(batch_data, tokenizer)
#
# # 创建 DataLoader
# from torch.utils.data import DataLoader
#
# dataloader = DataLoader(batch_dataset, batch_size=600)
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
# # 此时，hook.output 就是我们想要的层的输出
# pre_classifier_output = hook.output
# # print(pre_classifier_output)
# print('pre_classifier_output.shape', pre_classifier_output.shape)
# # 使用完钩子后，记得关闭它
# hook.close()
# pre_classifier_output = pre_classifier_output.cpu().numpy()
# # print(pre_classifier_output)
# print(pre_classifier_output.shape)
#
# # 标签矩阵
# label_matrix = np.zeros((600, 3))
# for i in range(600):
#     label_matrix[i][batch['labels'][i]] = 1
# # print(label_matrix)
# print(label_matrix.shape)
#
# # 计算权重矩阵
# weight_classifier = np.dot(np.linalg.pinv(pre_classifier_output), label_matrix)
# print("weight_classifier", weight_classifier)
# print(weight_classifier.shape)
#
# # 从训练数据中取出第二批数据
# hook = ForwardHook(model.classifier.fc9)
# batch_data1 = train_data.iloc[600:650]
# # batch_data1 = train_data.iloc[600:640]
# batch_dataset1 = DataFrameDataset(batch_data1, tokenizer)
# dataloader = DataLoader(batch_dataset1, batch_size=50)
#
# batch = next(iter(dataloader))
#
# # 将数据输入模型
# inputs = batch['input_ids'].to(device)
# attention_mask = batch['attention_mask'].to(device)
# labels = batch['labels'].to(device)
#
# with torch.no_grad():
#     model(inputs, attention_mask=attention_mask, labels=labels)
# pre_classifier_output1 = hook.output
# hook.close()
# pre_classifier_output1 = pre_classifier_output1.cpu().numpy()
# print(pre_classifier_output1.shape)
# out = np.dot(pre_classifier_output1, weight_classifier)
# print('out', out)
#
#
# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)
#
#
# # 对每一行应用 softmax 函数
# softmax_out = np.apply_along_axis(softmax, 1, out)
#
# print("softmax_out", softmax_out)
#
# # 假设 batch['labels'] 是真实的标签
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