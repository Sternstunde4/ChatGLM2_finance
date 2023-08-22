import re
from pathlib import Path

import pandas as pd
import numpy as np
import datasets
from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoConfig
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import transformers
import json
from sklearn.model_selection import train_test_split



data = pd.read_json('finance_en/finance_en.json',lines=True)
dftrain,dftest = train_test_split(data,test_size = 0.2)

#将上下文整理成与推理时候一致，参照model.chat中的源码~
#model.build_inputs??
def build_inputs(instruction, input):
    prompt = f"""
    请回答以下问题或执行相关操作：{instruction}
    """
    if input != "":
        prompt += f"输入：{input} "
    prompt += "答："
    return prompt


print(build_inputs('Describe the time management process for a project.',""))

dftrain['context'] = [build_inputs(x[0],x[1]) for x in dftrain.values]
dftrain['target'] = [x[2] for x in dftrain.values]
dftrain = dftrain[['context','target']]

dftest['context'] = [build_inputs(x[0],x[1]) for x in dftest.values]
dftest['target'] = [x[2] for x in dftest.values]
dftest = dftest[['context','target']]

ds_train = datasets.Dataset.from_pandas(dftrain)
ds_val = datasets.Dataset.from_pandas(dftest)



model_name = "D:\LLM_dev\LangChain\model\chatglm2-6b-32k"
max_seq_length = 512
skip_over_length = True

device = 'cuda:0'
model_path = re.sub("\s", "", "D:\LLM_dev\LangChain\model\chatglm2-6b-32k")
checkpoint = Path(f'{model_path}')
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, device=device)
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     model_name, trust_remote_code=True)

# config = transformers.AutoConfig.from_pretrained(
#     model_name, trust_remote_code=True, device_map='auto')
config = transformers.AutoConfig.from_pretrained(
    "D:\LLM_dev\LangChain\model\chatglm2-6b-32k", trust_remote_code=True, device=device)

def preprocess(example):
    context = example["context"]
    target = example["target"]

    context_ids = tokenizer.encode(
        context,
        max_length=max_seq_length,
        truncation=True)

    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)

    input_ids = context_ids + target_ids + [config.eos_token_id]

    return {"input_ids": input_ids, "context_len": len(context_ids), 'target_len': len(target_ids)}


ds_train_token = ds_train.map(preprocess).select_columns(['input_ids', 'context_len','target_len'])
if skip_over_length:
    ds_train_token = ds_train_token.filter(
        lambda example: example["context_len"]<max_seq_length and example["target_len"]<max_seq_length)


ds_val_token = ds_val.map(preprocess).select_columns(['input_ids', 'context_len','target_len'])
if skip_over_length:
    ds_val_token = ds_val_token.filter(
        lambda example: example["context_len"]<max_seq_length and example["target_len"]<max_seq_length)


def data_collator(features: list):
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

    input_ids = []
    labels_list = []

    for length, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        context_len = feature["context_len"]

        labels = (
                [-100] * (context_len - 1) + ids[(context_len - 1):] + [-100] * (longest - length)
        )  # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss

        ids = ids + [tokenizer.pad_token_id] * (longest - length)

        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labels))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

dl_train = torch.utils.data.DataLoader(ds_train_token,num_workers=0,batch_size=4,  # num_workers原来=2，因为在linux系统中可以使用多个子进程加载数据，而在windows系统中不能
                                       pin_memory=True,shuffle=True,
                                       collate_fn = data_collator)
dl_val = torch.utils.data.DataLoader(ds_val_token,num_workers=0,batch_size=4,
                                    pin_memory=True,shuffle=True,
                                     collate_fn = data_collator)

for batch in dl_train:
    break

dl_train.size = 300 #每300个step视作一个epoch，做一次验证
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, device=device).half().cuda()
model.eval()
# model = AutoModel.from_pretrained("D:\LLM_dev\LangChain\model\chatglm2-6b",
#                                   load_in_8bit=False,
#                                   trust_remote_code=True,
#                                   device='auto')

model.supports_gradient_checkpointing = True  #节约cuda
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
#model.lm_head = CastOutputToFloat(model.lm_head)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
model.is_parallelizable = True
model.model_parallel = True
model.print_trainable_parameters()
