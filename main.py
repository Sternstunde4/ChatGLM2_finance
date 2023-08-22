# -*- coding: utf-8 -*-

from modelscope.msdatasets import MsDataset


# 数据探索，使用流式方式查看
# ds = MsDataset.load('chatglm_llm_fintech_raw_dataset', split='train', use_streaming=True)
# print(next(iter(ds)))

# 加载全量数据(大小56GB，可能会加载较长时间)
# 每条数据为pdf文件的本地缓存路径
# ds = MsDataset.load('chatglm_llm_fintech_raw_dataset', split='train', cache_dir='D:\LLM_dev\FinGPT-intern\dataset' ,encoding='gbk')
# print(next(iter(ds)))
ds = MsDataset.load('chatglm_llm_fintech_raw_dataset', split='train', use_streaming=True, stream_batch_size=1, cache_dir='D:\LLM_dev\FinGPT-intern\dataset')
for item in ds:
    print(item)

# 备注: 
# 1. 自定义缓存路径，可以自行设置cache_dir参数，即 MsDataset.load(..., cache_dir='/to/your/path')
# 2. 补充数据加载（从9493条增加到11588条），sdk加载注意事项
    #  a) 删除缓存中的csv映射文件(默认路径为)： ~/.cache/modelscope/hub/datasets/modelscope/chatglm_llm_fintech_raw_dataset/master/data_files/732dc4f3b18fc52380371636931af4c8
    #  b) 使用MsDataset.load(...) 加载，默认会reuse已下载过的文件，不会重复下载。


# 加载结果示例（单条，pdf:FILE字段值为该pdf文件本地缓存路径，文件名做了SHA转码，可以直接打开） 
# {'name': '2020-03-24__北京鼎汉技术集团股份有限公司__300011__鼎汉技术__2019年__年度报告.pdf',
# 'pdf:FILE': '~/.cache/modelscope/hub/datasets/modelscope/chatglm_llm_fintech_raw_dataset/master/data_files/430da7c46fb80d4d095a57b4fb223258ffa1afe8bf53d0484e3f2650f5904b5c'}