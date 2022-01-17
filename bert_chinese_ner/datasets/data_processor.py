import os
import json
import importlib
import opencc

import bert_chinese_ner
from bert_chinese_ner import config

print(config)


importlib.reload(bert_chinese_ner)
importlib.reload(config)

os.getcwd()


file_path = valid_path


converter = opencc.OpenCC('s2t.json')
converter.convert('汉字')


path = config.ROOT_DIR
train_file = config.TRAINING_FILE
dev_file = config.DEV_FILE


file_path = dev_path

aa = ['在 O', '这 O', '里 O', '恕 O', '弟 O', '不 O', '恭 O']
aa = ['在 O', '这 O', '里 O', '恕 O', '弟 O', '不 O', '恭 O', '之 O', '罪 O', '， O', '敢 O', '在 O', '尊 O', '前 O', '一 O', '诤 O', '： O', '前 O', '人 O', '论 O', '书 O', '， O', '每 O', '曰 O', '“ O', '字 O', '字 O', '有 O', '来 O', '历 O', '， O', '笔 O', '笔 O', '有 O', '出 O', '处 O', '” O', '， O', '细 O', '读 O', '公 O', '字 O', '， O', '何 O', '尝 O', '跳 O', '出 O', '前 O', '人 O', '藩 O', '篱 O', '， O', '自 O', '隶 O', '变 O', '而 O', '后 O', '， O', '直 O', '至 O', '明 O', '季 O', '， O', '兄 O', '有 O', '何 O', '新 O', '出 O', '？ O', '', '相 O', '比 O', '之 O', '下 O', '， O', '青 B-ORG', '岛 I-ORG', '海 I-ORG', '牛 I-ORG', '队 I-ORG', '和 O', '广 B-ORG', '州 I-ORG', '松 I-ORG', '日 I-ORG', '队 I-ORG', '的 O', '雨 O', '中 O', '之 O', '战 O', '虽 O', '然 O', '也 O', '是 O', '0 O', '∶ O', '0 O', '， O', '但 O', '乏 O', '善 O', '可 O', '陈 O', '。 O', '', '理 O', '由 O', '多 O', '多 O', '， O', '最 O', '无 O', '奈 O', '的 O', '却 O', '是 O', '： O', '5 O', '月 O', '恰 O', '逢 O', '双 O', '重 O', '考 O', '试 O', '， O', '她 O', '攻 O', '读 O', '的 O', '博 O', '士 O', '学 O', '位 O', '论 O', '文 O', '要 O', '通 O', '考 O', '； O', '她 O', '任 O', '教 O', '的 O', '两 O', '所 O', '学 O', '校 O', '， O', '也 O', '要 O', '在 O', '这 O', '段 O', '时 O', '日 O', '大 O', '考 O', '。 O', '', '分 O', '工 O', '， O', '各 O', '有 O', '各 O', '的 O', '责 O', '任 O', '； O', '合 O', '作 O', '， O', '正 O', '副 O', '经 O', '理 O', '之 O', '间 O', '， O', '全 O', '厂 O', '的 O', '事 O', '， O', '不 O', '管 O', '由 O', '谁 O', '分 O', '管 O', '， O', '也 O', '不 O', '管 O', '你 O', '有 O', '什 O', '么 O', '事 O', '找 O', '到 O', '谁 O', '， O', '绝 O', '不 O', '会 O', '把 O', '你 O', '推 O', '给 O', '第 O', '二 O', '个 O', '人 O', '。 O', '', '胡 B-PER', '老 O', '说 O', '， O', '当 O', '画 O', '画 O', '疲 O', '倦 O', '时 O', '就 O', '到 O', '院 O', '里 O', '去 O', '看 O', '看 O', '， O', '给 O', '这 O', '盆 O', '花 O', '浇 O', '点 O', '水 O', '， O', '给 O', '那 O', '棵 O', '花 O', '剪 O', '剪 O', '枝 O', '， O', '回 O', '来 O', '再 O', '接 O', '着 O', '画 O', '， O', '画 O', '累 O', '了 O', '再 O', '出 O', '去 O', '， O', '如 O', '此 O', '循 O', '环 O', '往 O', '复 O', '， O', '脑 O', '体 O', '结 O', '合 O']
len(aa)
aa.split()

('\n'.join(aa)).split('\n')






def _parse_data(file_path, text_index=0, label_index=1):
    x_data = [], y_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        lines = converter.convert('\n'.join(lines[:100])).split('\n')
        # lines: ['在 O', '這 O', ..., '罪 O']
        print(lines, '\n\n')
        one_samp_x, one_samp_y = [], []
        for line in lines:
            # '在 O' -> ['在', 'O'], '這 O' -> ['這', 'O'],...
            row = line.split(' ')
            if len(row) == 1:
                x_data.append(x) # Add a complete sample to x_data
                y_data.append(y)
                one_samp_x = [] # reset the sample container
                one_samp_y = []
            else:
                one_samp_x.append(row[text_index]) # extend the sample with a char
                one_samp_y.append(row[label_index])
    return x_data, y_data



def load_data(path=None, train_file=None, dev_file=None):
    train_path = os.path.join(path, train_file)
    dev_path = os.path.join(path, dev_file)

    train = _parse_data(train_path)
    dev = _parse_data(dev_path)

