import json
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from utils import get_answer, parse_between

# 데이터 로드
trainfp = '../data/ML-ESG-3_Trainset_Korean.json'
testfp = '../data/ML-ESG3_Testset_Korean.json'

with open(trainfp, 'r') as f:
    train = json.load(f)

with open(testfp, 'r') as f:
    test = json.load(f)

# tmap 정의
tmap = {
    'short': 'less than 2 years',
    'medium': '2 to 5 years',
    'long': 'more than 5 years',
}
tmap_reverse = {v: k for k, v in tmap.items()}

# 데이터 프레임 생성
df = pd.DataFrame(train)[['url', 'category', 'title', 'content', 'impact_type', 'impact_duration']]

# BM25 초기화
corpus = [item['content'] for item in train]
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# 프롬프트 정의
guidelines = """
You will be given a text. Refer to the examples for your decision. Classify it to either [short/medium/long] based on its expected timeframe.
Refer to the following guidelines. Answer in the format: "The answer is <ans>".

### Guidelines:
Long:
- Long-Term Event Risk: Longer-term physical impacts of climate change could cause operational disruptions in the long run.
- Growing Stakeholder Awareness: Issue receives scrutiny fromspecialized or niche stakeholders, increasing coverage in media and growing presence in public eye.
- Long-Term Resource Scarcity: Key resource/input is constrained, which
may lead to cost increase or operational disruption in 5+ years.
- Forecast Demand Shift: Underlying social or environmental pressures (e.g., obese population, climat change) are likely t cause change in demand and consumer preference over time.
- Competitor Response: Major industry player(s) undertake strategic response to environmental or social trends, increasing competitive pressure.

Medium: 
- Emerging Event Risk: Operational even could threaten ability to grow or license to operate, but will likely manifest over extended time frame (e.g., mounting community opposition to a project; major investigations, settlements, or prosecution)
- Emerging Regulatory Pressure: Issue receives increasing scrutiny from mainstream stakeholders; pressure on regulators is mounting but no pending regulatory change yet.
- Medium-Term Supply Constraint: Supply constraint is forecast, could cause disruption or significant cost increase in 2- to 5-year period.
- Demonstrated Demand Shift: Growing demand in “new” area evidenced by differential growth rates and notable shift in market share between substitutes. Incentive-Based
- Demand Shift: Demand shift over 2- to 5-year period will be led by government incentives

Short:
- Acute Event Risk: Sudden operational event could limit company’s ability to
grow (e.g., exploit new reserves, expand to new territories), cause significant liabilities, disrupt key business units, or threaten overall business model
- Imminent Regulator Change: Regulatory change is pending or imminent in key markets.
- Short-Term Supply Constraint: Acute supply constraint is present or imminent, likely to cause disruption or significant cost increase for companies.
- Acute Demand Shift: Ongoing demand shift between substitute products/services, old product/service becoming obsolete.
"""

# 결과 저장
output = []
for item in tqdm(test):
    tokenized_query = item['content'].split(" ")
    candidates = bm25.get_top_n(tokenized_query, corpus, n=5)
    batch = df[df.content.isin(candidates)].iloc[::-1]

    # In-Context Learning을 통한 예제 추가
    icl = "\n\n".join([
        f"### text: {t}\n### response: Based on the MSCI guidelines, I classify this to <{tmap_reverse[i]}>."
        for t, i in batch[['content', 'impact_duration']].values
    ])
    prompt = f"{guidelines}\n\n{icl}\n\n### text: {item['content']}\n### response:"

    # GPT-4 호출 및 결과 저장
    query, response = get_answer(prompt, item['content'], "gpt-4-0125-preview")
    impact_duration = parse_between(response, "<", ">")
    output.append([query, response, impact_duration])

# 결과 데이터프레임 생성 및 저장
df_output = pd.DataFrame(output, columns=['prompt', 'generated', 'impact_duration'])
df_output.to_csv('../results/impact_duration.csv', index=False)