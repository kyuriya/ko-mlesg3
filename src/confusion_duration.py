import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 파일 경로 설정
testfp = '../data/ML-ESG3_Testset_Korean.json' # Gold Standard 데이터
gpt4_fp = '../data/Korean_FIT_1.json'  # GPT-4 결과 데이터

# 데이터 로드
with open(gpt4_fp, 'r') as file:
    gpt4_data = json.load(file)

with open(testfp, 'r') as file:
    gold_data = json.load(file)

# GPT-4 결과와 Gold Standard 비교
output = []
for gpt4, gold in zip(gpt4_data, gold_data):
    row = [
        gpt4['url'],
        gpt4['category'],
        gpt4['impact_duration'],
        gold['impact_duration']
    ]
    output.append(row)


columns = ['url', 'category', 'duration_gpt4', 'duration_gold']
df = pd.DataFrame(output, columns=columns)

# 데이터 전처리
df['duration_gpt4'] = df['duration_gpt4'].str.capitalize()
df['duration_gold'] = df['duration_gold'].str.capitalize()

contingency_duration = pd.crosstab(df['duration_gpt4'], df['duration_gold'], normalize='all')

# confusion matrix 시각화
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.1) 
sns.heatmap(contingency_duration, annot=True, cmap='Blues', fmt='.2%', cbar_kws={'label': 'Percentage'})

plt.xlabel('Predicted (GPT-4)', fontsize=12)
plt.ylabel('Actual (Gold Standard)', fontsize=12)
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)

# confusion matrix 그래프 저장 경로
cm_duration_fp = '../results/cm_duration.pdf'  

# PDF로 저장
plt.savefig(cm_duration_fp, bbox_inches='tight', dpi=300)
plt.close()
