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

# 每epoch从transformer内部获取数据，最终求和输入到pre_classifier同时lora训练 ×
# 将训练好的模型参数读取，进行低秩LoRA训练 √

os.environ["TOKENIZERS_PARALLELISM"] = "true"
'''--------------------model load----------------------'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './distilbert-base-multilingual-cased-sentiments-student/'
config = AutoConfig.from_pretrained("./distilbert-base-multilingual-cased-sentiments-student/config.json")
tokenizer_path = './distilbert-base-multilingual-cased-sentiments-student/'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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


lora_config = LoraConfig(
    task_type="SEQUENCE_CLASSIFICATION",
    inference_mode=False,
    r=256,
    lora_alpha=256,
    lora_dropout=0.1,
    target_modules=[
        "transformer.layer.0.attention.q_lin",
        "transformer.layer.0.attention.k_lin",
        "transformer.layer.0.attention.v_lin",
        "transformer.layer.0.attention.out_lin",
        "transformer.layer.0.ffn.lin1",
        "transformer.layer.0.ffn.lin2",
        "transformer.layer.1.attention.q_lin",
        "transformer.layer.1.attention.k_lin",
        "transformer.layer.1.attention.v_lin",
        "transformer.layer.1.attention.out_lin",
        "transformer.layer.1.ffn.lin1",
        "transformer.layer.1.ffn.lin2",
        "transformer.layer.2.attention.q_lin",
        "transformer.layer.2.attention.k_lin",
        "transformer.layer.2.attention.v_lin",
        "transformer.layer.2.attention.out_lin",
        "transformer.layer.2.ffn.lin1",
        "transformer.layer.2.ffn.lin2",
        "transformer.layer.3.attention.q_lin",
        "transformer.layer.3.attention.k_lin",
        "transformer.layer.3.attention.v_lin",
        "transformer.layer.3.attention.out_lin",
        "transformer.layer.3.ffn.lin1",
        "transformer.layer.3.ffn.lin2",
        "transformer.layer.4.attention.q_lin",
        "transformer.layer.4.attention.k_lin",
        "transformer.layer.4.attention.v_lin",
        "transformer.layer.4.attention.out_lin",
        "transformer.layer.4.ffn.lin1",
        "transformer.layer.4.ffn.lin2",
        "transformer.layer.5.attention.q_lin",
        "transformer.layer.5.attention.k_lin",
        "transformer.layer.5.attention.v_lin",
        "transformer.layer.5.attention.out_lin",
        "transformer.layer.5.ffn.lin1",
        "transformer.layer.5.ffn.lin2",
        "pre_classifier",
        "classifier"
    ]
)

model = get_peft_model(model, lora_config)

print(model.get_nb_trainable_parameters())

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

# for name, param in model.named_parameters():
#     # param.requires_grad = False
#     if 'fc_2out_lin' in name or 'fc_3out_lin' in name or 'fc_4out_lin' in name:
#         param.requires_grad = True
# # 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)

for i in range(6):
    print("training...")
    trainer.train()
    # model.save_pretrained("./output/sequenceTune/model/")
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


