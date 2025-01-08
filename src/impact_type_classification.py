import json
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from utils import get_answer, parse_between

# 데이터 로드
trainfp = '../dataset/ML-ESG-3_Trainset_Korean.json'
testfp = '../dataset/ML-ESG3_Testset_Korean.json'

with open(trainfp, 'r') as f:
    train = json.load(f)

with open(testfp, 'r') as f:
    test = json.load(f)

# 데이터 프레임 생성
df = pd.DataFrame(train)[['url', 'category', 'title', 'content', 'impact_type', 'impact_duration']]

# BM25 초기화
corpus = [item['content'] for item in train]
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# 결과 저장
output = []
for item in tqdm(test):
    tokenized_query = item['content'].split(" ")
    candidates = bm25.get_top_n(tokenized_query, corpus, n=5)
    batch = df[df.content.isin(candidates)].iloc[::-1]

    # 프롬프트 생성
    examples = "\n".join([f"### text:{t}\n### response: Based on the MSCI guidelines, I classify this to <{i}>."
                          for t, i in batch[['content', 'impact_type']].values])
    prompt = (
        "You will be given a text. Refer to the examples for your decision. "
        "Classify it to either [cannot distinguish/risk/opportunity] based on the impact it will have on the company.\n"
    )
    prompt += examples
    prompt += f"\n### text: {item['content']}\n### response:"

    # GPT-4 호출 및 결과 저장
    o = get_answer(prompt, item['content'], "gpt-4-0125-preview")
    output.append([o[0], o[1], parse_between(o[1], '<', '>')])

# 결과 CSV 저장
df_output = pd.DataFrame(output, columns=['prompt', 'generated', 'impact_type'])
df_output.to_csv('../results/impact_type.csv', index=False)