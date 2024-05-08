import pyarrow.parquet as pq
import pandas as pd
import torch
from torch.utils.data import dataset
from transformers import pipeline, Trainer
from transformers import AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType,PeftConfig
from torch.utils.data import Dataset
import torch.nn.functional as F

#不训练，只加载模型做推理，计算得分并汇总到表

'''--------------------model load----------------------'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './output/LoRA_test_trainer/model'
# model_path = './distilbert-base-multilingual-cased-sentiments-student/'
# config = PeftConfig.from_pretrained("./output/LoRA_test_trainer/model/adapter_config.json")
tokenizer_path = './distilbert-base-multilingual-cased-sentiments-student/'

model = AutoModelForSequenceClassification.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print(model)


# parquet_file = pq.ParquetFile('./Datasets/glue/train-00000-of-00001.parquet')
# data = parquet_file.read().to_pandas()
data = pd.read_excel('./Datasets/glue/train_data_result.xlsx')
data['new result'] = ''
data['new score'] = 0.0
data = data.iloc[0:1000]
# print(data)
# data.to_excel('./Datasets/glue/train_data.xlsx')

'''--------------------pipeline----------------------'''
# for i in range(len(data)):
#     # print(data.iloc[i]['sentence'])
#     # print(data.iloc[i]['label'])
#     print(data.iloc[i]['idx'])
#     print('-----------------')
#     sentiment_analysis = pipeline('sentiment-analysis', model=model_path, tokenizer=tokenizer_path)
#     analysis_result=sentiment_analysis(data.iloc[i]['sentence'])
#     data.at[i, 'new result'] = analysis_result[0]['label']
#     data.at[i, 'new score'] = analysis_result[0]['score']
#     print(analysis_result)
# data.to_excel('./Datasets/glue/LoRA_trained_result.xlsx')

#原数据集0和1标反，train_data和validation_data已更正，new_trained_result为全局调整、未更正的结果


'''--------------------Inference----------------------'''
# for i in range(len(data)):
#     # 假设我们有一个句子
#     sentence = data.iloc[i]['sentence']
#     # 使用tokenizer将文本转换为模型可以理解的格式
#     inputs = tokenizer(sentence, padding="max_length", truncation=True, return_tensors="pt")
#     # 将输入数据移动到相同的设备上
#     inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
#     # 使用模型进行推理
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # 输出是一个元组，其中第一个元素是logits
#     logits = outputs[0]
#     # 我们可以使用argmax来获取最可能的类别的索引
#     predicted_class_idx = logits.argmax(-1).item()
#     # 使用softmax函数将logits转换为概率分布
#     probabilities = F.softmax(logits, dim=-1)
#     # 获取我们感兴趣的类别的概率作为评分
#     score = probabilities[0, predicted_class_idx].item()
#     # 获取模型的类别标签
#     labels = model.config.id2label
#     # 获取预测的类别
#     predicted_class = labels[predicted_class_idx]
#     print(f"Predicted class: {predicted_class}, Score: {score}")
#     data.at[i, 'new result'] = predicted_class
#     data.at[i, 'new score'] = score
#
# data.to_excel('./Datasets/glue/freeze_trained_result.xlsx')
