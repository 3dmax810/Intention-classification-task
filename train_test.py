import json
import numpy as np
import os
import random
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# label_map = {
#     "闲聊": 0,
#     "知识问答": 1,
#     "讲故事或创作故事": 2,
#     "背古诗": 3,
#     "英语对话": 4,
#     "查天气": 5,
#     "结束聊天的想法": 6,
#     "切换角色": 7,
#     "翻译或使用另一种语言比如英文等形容": 8,
#     "打招呼": 9,
#     "需要呵护或关心": 10,
#     "被用户夸奖": 11,
#     "听歌": 12,
# }


# 数据集制作
def get_datasets(json_dir="", json_path=""):
    json_files = []  # json 内容
    for file in os.listdir(json_dir):
        if file.endswith(".json"):
            json_files.append(os.path.join(json_dir, file))

    train_dataset = []

    # print(json_files[0:1])
    for file in json_files:
        print(file)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            train_dataset.extend(data[:int(len(data))])
    random.shuffle(train_dataset)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f)
    print(True)


# get_datasets()


def make_dataset(json_dir):
    with open(json_dir, "r", encoding="utf-8") as f:
        data = json.load(f)
        results = []
        for item in data:
            context = item['context']
            parts = []
            for i, turn in enumerate(context):
                parts.append(f"用户: {turn['user']}")
                if i < len(context) - 1:  # 最后一轮 bot 留空就不加了
                    parts.append(f"系统: {turn['bot']}")
            merged_text = " <SEP> ".join(parts)
            results.append({
                "text": merged_text,
                "label": item['intent_id']
            })
        return results


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }


def train(json_dir="",
          model_path="",
          save_path="/"):
    with open(json_dir, "r", encoding="utf-8") as f:
        data = make_dataset(json_dir)
    train_data, eval_data = train_test_split(data, test_size=0.01, random_state=42)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=13)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    def preprocess(results):
        encoding = tokenizer(
            results["text"],  # 输入字段，如 query 或上下文拼接文本
            padding="max_length",
            truncation=True,
            max_length=128,
            # return_tensors="pt"
        )
        encoding["label"] = results["label"]  # 加入标签字段（模型训练需要用到）
        return encoding

    train_dataset = Dataset.from_list(train_data).map(preprocess)
    eval_dataset = Dataset.from_list(eval_data).map(preprocess)

    training_args = TrainingArguments(
        output_dir="../results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        logging_dir="../logs",
        save_strategy="epoch",
        load_best_model_at_end=True,  # 自动加载最好的模型
        metric_for_best_model="accuracy",
        logging_steps=100,
        save_total_limit=2,  # 限制最多保存两个 checkpoint
        learning_rate=2e-5,
        warmup_steps=500,  # 预热步数
        weight_decay=0.01,
        # 验证时的 batch size
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)


# train()


def user_text(model_path=""):
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=13)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.eval()
    while True:
        query = input("shuru")
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()

        print("预测的 intent_id:", predicted_class+1)

# user_text()
