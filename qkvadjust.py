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

# 从transformer内部获取数据，最终求和输入到pre_classifier同时lora训练

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
        self.fc_2out_lin = nn.Linear(768, 768)
        self.fc_3out_lin = nn.Linear(768, 768)
        self.fc_4out_lin = nn.Linear(768, 768)
        # self.original_model.pre_classifier.register_forward_hook(self.pre_classifier_hook)
        # self.original_model.distilbert.transformer.layer[5].register_forward_hook(self.transformer_hook)
        self.original_model.distilbert.transformer.layer[1].attention.out_lin.register_forward_hook(self.fc2_hook)
        self.original_model.distilbert.transformer.layer[2].attention.out_lin.register_forward_hook(self.fc3_hook)
        self.original_model.distilbert.transformer.layer[3].attention.out_lin.register_forward_hook(self.fc4_hook)

    # def pre_classifier_hook(self, module, input, output):
    #     self.pre_classifier_output = output

    # def transformer_hook(self, module, input, output):
    #     self.transformer_output = output

    def fc2_hook(self, module, input, output):
        self.fc2output = output

    def fc3_hook(self, module, input, output):
        self.fc3output = output

    def fc4_hook(self, module, input, output):
        self.fc4output = output

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                      labels=labels)
        hidden_states = outputs.hidden_states

        transformer6_output = hidden_states[6] + self.fc2output + self.fc3output + self.fc4output

        transformer6_output = transformer6_output[:, 0, :]

        # 旁路和最后一个transformer的输出输入到pre_classifier
        pre_classifier_input = transformer6_output
        pre_classifier_output = self.original_model.pre_classifier(pre_classifier_input)
        # 将全连接层的输出和pre_classifier的输出一起输入到classifier中
        classifier_input = pre_classifier_output
        # classifier_input = fc_4_output +fc_5_output+fc_3_output+fc_2_output+ pre_classifier_output
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


model = ModifiedModel(model)

print(model)
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

# def tokenize_function(examples):
#     encoded = tokenizer(examples['sentence'], padding="max_length", truncation=True)
#     if 'input_ids' not in encoded or len(encoded['input_ids']) == 0:
#         raise ValueError(f"Empty input_ids for sentence {examples['sentence']}")
#     return encoded
#
# train_dataset=train_data.map(tokenize_function, batched=True)
# test_dataset=test_data.map(tokenize_function, batched=True)


lora_config = LoraConfig(
    task_type="SEQUENCE_CLASSIFICATION",
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "transformer.layer.2.attention.out_lin",
        "transformer.layer.3.attention.q_lin",
        "transformer.layer.3.attention.k_lin",
        "transformer.layer.3.attention.v_lin",
        "transformer.layer.3.attention.out_lin",
        "transformer.layer.4.attention.q_lin",
        "transformer.layer.4.attention.k_lin",
        "transformer.layer.4.attention.v_lin",
        "transformer.layer.4.attention.out_lin",
        "transformer.layer.5.attention.q_lin",
        "transformer.layer.5.attention.k_lin",
        "transformer.layer.5.attention.v_lin",
        "transformer.layer.5.attention.out_lin",
    ]
)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits[0].tolist()  # 使用logits元组的第一个元素
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


model = get_peft_model(model, lora_config)

print(model.get_nb_trainable_parameters())

training_args = TrainingArguments(
    output_dir="./output/temp_trainer",
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    save_strategy="steps",  # 设置为"steps"
    save_steps=1e10,  # 设置为一个非常大的数
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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

for name, param in model.named_parameters():
    # param.requires_grad = False
    if 'fc_2out_lin' in name or 'fc_3out_lin' in name or 'fc_4out_lin' in name:
        param.requires_grad = True

# # 打印出所有需要训练的参数
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)

print("training...")
trainer.train()

'''--------------------model eval----------------------'''

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


