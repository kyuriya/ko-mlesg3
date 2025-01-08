import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 파일 경로 설정
testfp = '../dataset/ML-ESG3_Testset_Korean.json'  # Gold Standard 데이터
gpt4_fp = '../dataset/Korean_FIT_1.json'  # GPT-4 결과 데이터


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
        gpt4['impact_type'],
        gold['impact_type']
    ]
    output.append(row)


columns = ['url', 'category', 'type_gpt4', 'type_gold']
df = pd.DataFrame(output, columns=columns)

# 데이터 전처리
df['type_gpt4'] = df['type_gpt4'].str.capitalize()
df['type_gold'] = df['type_gold'].str.capitalize()

# Impact Type confusion matrix
contingency_type = pd.crosstab(df['type_gpt4'], df['type_gold'], normalize='all')

# 시각화
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.1)  
sns.heatmap(contingency_type, annot=True, cmap='Blues', fmt='.2%', cbar_kws={'label': 'Percentage'})

plt.xlabel('Predicted (GPT-4)', fontsize=12)
plt.ylabel('Actual (Gold Standard)', fontsize=12)
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)

# confusion matrix 그래프 저장 경로
cm_type_fp = '../results/cm_type.pdf'  

# PDF로 저장
plt.savefig(cm_type_fp, bbox_inches='tight', dpi=300)
plt.close()