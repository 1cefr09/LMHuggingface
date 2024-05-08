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
        self.original_model = original_model
        self.fc_4 = nn.Linear(768, 768)  # 假设transformer.layer.4的输出维度为768

        self.original_model.pre_classifier.register_forward_hook(self.pre_classifier_hook)
        self.original_model.distilbert.transformer.layer[4].register_forward_hook(self.transformer_hook)

    def pre_classifier_hook(self, module, input, output):
        self.pre_classifier_output = output

    def transformer_hook(self, module, input, output):
        self.transformer4_output = output

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                      labels=labels)
        hidden_states = outputs.hidden_states
        # 获取transformer.layer.4的输出
        intermediate_output4 = hidden_states[5]
        # intermediate_output41 = self.transformer4_output
        # print("intermediate_output4:", intermediate_output4)
        # print("intermediate_output41:", intermediate_output41)
        # 将transformer.layer.4的输出作为全连接层的输入
        intermediate_output4 = intermediate_output4[:, 0, :]
        fc_4_output = self.fc_4(intermediate_output4)
        # print("fc_4_output_shape:", fc_4_output.shape)
        # 获取pre_classifier的输出
        pre_classifier_output = self.pre_classifier_output
        # print("pre_classifier_output_shape:", pre_classifier_output.shape)
        # 将全连接层的输出和pre_classifier的输出一起输入到classifier中
        # classifier_input = pre_classifier_output
        classifier_input = fc_4_output + pre_classifier_output
        # print("classifier_input_shape:", classifier_input.shape)
        classifier_output = self.original_model.classifier(classifier_input)
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

model=ModifiedModel(model)
model.to(device)
print(model)
checkpoint_path = "./output/freeze_trainer/checkpoint-2106"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
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

for name, param in model.named_parameters():
    param.requires_grad = False
    if 'fc_4' in name:
        param.requires_grad = True
# # 打印出所有需要训练的参数
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
    logits = logits[0].tolist()  # 使用logits元组的第一个元素
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
                  output_dir="./output/freeze_trainer",
                  evaluation_strategy="epoch",
                  # save_strategy="epoch",
                  save_strategy="epoch",  # 设置为"steps"
                  # save_steps=1e10,  # 设置为一个非常大的数
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
trainer.train(resume_from_checkpoint=checkpoint_path)


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