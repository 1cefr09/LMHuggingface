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
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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
    return metric.compute(predictions=predictions, references=labels)
from transformers.modeling_outputs import SequenceClassifierOutput

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
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1)
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
    if 'classifier' in name:
        param.requires_grad = True
# 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
print("training...")
trainer.train()
torch.save(model.state_dict(), './output/sequenceTune/model.pth')

'''--------------------model eval----------------------'''
print("evaluating...")
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
