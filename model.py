import json
import os
import pickle

from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
import re
from pathlib import Path
from tqdm import *


device = 'cuda:0'
model_path = re.sub("\s", "", "D:/LLM_dev/LangChain/model/chatglm2-6b-32k")
checkpoint = Path(f'{model_path}')
# print(checkpoint)

embeddings = HuggingFaceEmbeddings(model_name="D:/LLM_dev/LangChain/model/bge-large-zh", model_kwargs={'device': "cuda"})
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, device=device)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, device=device).half().cuda()
model = model.eval()

# max_history_length = 900  # 2048token
global history
history = []
response, history = model.chat(tokenizer, "你好", history=history)
print(response)

# 获取公司_txt.txt对应的序号作为向量数据库的名称
with open('vector_store/dict.pkl', 'rb') as f:
    dict = pickle.load(f)


with open('result_demo\\test_questions.json', 'r', encoding='utf-8') as file:
    json_data = [json.loads(line) for line in file]

for id,item in tqdm(enumerate(json_data)):
    if id < 4615:
        continue
    print(id)
    found = 0
    knowledge = ''
    match_keys = []

    for key, value in dict.items():
        #  获取主体名称和简称
        pattern1 = r'__(.*?)__'
        matches3 = re.findall(pattern1, value)
        entity = []
        for match3 in matches3:
            entity.append(match3)
        matches1 = re.findall(rf"{entity[0]}", item["question"], re.MULTILINE)
        matches2 = re.findall(rf"{entity[1]}", item["question"], re.MULTILINE)
        if (len(matches1) > 0) | (len(matches2) > 0):
            match_keys.append(key)
    current_dir = os.getcwd()
    if len(match_keys) > 0:
        path = os.path.join(current_dir + '\\vector_store0810', str(match_keys[0]))
        try:
            cp_db = FAISS.load_local(path, embeddings)
            found = 1
        except:
            pass
        if len(match_keys) > 1:  # 如该公司有多个年报，将所有数据库合并
            for key in match_keys[1:]:
                path = os.path.join(current_dir + '\\vector_store0810', str(key))
                try:
                    new_cp_db = FAISS.load_local(path, embeddings)
                    cp_db.merge_from(new_cp_db)
                    found = 1
                except:
                    # found = 0
                    pass
        masked_question = item["question"].replace(entity[0], "公司")  # 隐去公司全名
        masked_question = item["question"].replace(entity[1], "公司")  # 隐去公司简称
        contexts = cp_db.similarity_search(masked_question)
        count = 0
        for context in contexts:
            if count < 5:
                knowledge = knowledge + '\n' + context.page_content
                count += 1
            else:
                break


    if found == 0:
        # 通用金融问题
        prompt = f"""你是一个金融领域的专家，请回答以下金融问题：{item["question"]}
                    """
    else:
        prompt = f"""'''限定范围内的内容是与问题最相关的知识，其中可能含有表格，需要特别处理，表格的每一行由[]限定，其中每一列用逗号隔开，已知：'''{knowledge}'''，请根据'''限定范围内的内容，回答以下金融问题：{item["question"]}
        如果涉及到计算，请写明计算公式及逻辑推导过程，计算结果中的数字需要去掉其中的逗号，可参考如下常见的计算公式：
        企业研发经费与利润比值=研发费用/净利润
        企业研发经费与营业收入比值=研发费用/营业收入
        研发人员占职工人数比例=研发人员数/职工总数
        流动比率=流动资产/流动负债
        速动比率=(流动资产-存货)/流动负债
        企业硕士及以上人员占职工人数比例=(硕士人数 + 博士及以上人数)/职工总数
        企业研发经费占费用比例=研发费用/(销售费用+财务费用+管理费用+研发费用
        营业利润率=营业利润/营业收入
        资产负债比率=总负债/资产总额
        现金比率=货币资金/流动负债
        非流动负债比率=非流动负债/总负债
        流动负债比率=流动负债/总负债
        净资产收益率=净利润/净资产
        净利润率=净利润/营业收入
        营业成本率=营业成本/营业收入
        管理费用率=管理费用/营业收入
        财务费用率=财务费用/营业收入
        毛利率=(营业收入-营业成本)/营业收入
        净资产增长率=(净资产-上年净资产)/上年净资产
        三费比重=(销售费用+管理费用+财务费用)/营业收入
        投资收益占营业收入比率=投资收益/营业收入
        销售费用增长率=(销售费用-上年销售费用)/上年销售费用
        财务费用增长率=(财务费用-上年财务费用)/上年财务费用
        管理费用增长率=(管理费用-上年管理费用)/上年管理费用
        研发费用增长率=(研发费用-上年研发费用)/上年研发费用
        总负债增长率=(总负债-上年总负债)/上年总负债
        流动负债增长率=(流动负债-上年流动负债)/上年流动负债
        货币资金增长率=(货币资金-上年货币资金)/上年货币资金
        固定资产增长率=(固定资产-上年固定资产)/上年固定资产
        无形资产增长率=(无形资产-上年无形资产)/上年无形资产
        总资产增长率=(资产总额-上年资产总额)/上年资产总额
        营业收入增长率=(营业收入-上年营业收入)/上年营业收入
        营业利润增长率=(营业利润-上年营业利润)/上年营业利润
        净利润增长率=(净利润-上年净利润)/上年净利润
        现金及现金等价物增长率=(现金及现金等价物-上年现金及现金等价物)/上年现金及现金等价物
        """
    # print(prompt)
    Prompt = {}
    Prompt['id'] = item['id']
    Prompt['prompt'] = prompt
    with open('result_demo\prompts.json', 'a', encoding='utf-8') as file:
        json.dump(Prompt, file, ensure_ascii=False)
        file.write('\n')
    history = []
    try:
        response, history = model.chat(tokenizer, prompt, history=history)
        item["answer"] = response
        # print(response)
        with open('result_demo\\test_answers_0810.json', 'a', encoding='utf-8') as file:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')
    except:
        item["answer"] = ""
        with open('result_demo\\test_answers_0810.json', 'a', encoding='utf-8') as file:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')
        pass
