import pandas as pd
import json

# 파일 경로
impact_duration_fp = '../results/gpt4-classification/impact_duration.csv'
impact_type_fp = '../results/gpt4-classification/impact_type.csv'

impact_duration_df = pd.read_csv(impact_duration_fp)
impact_type_df = pd.read_csv(impact_type_fp)

# 두 결과를 병합
# Note: 'prompt' 컬럼을 기준으로 병합
merged_df = pd.merge(
    impact_duration_df[['prompt', 'impact_duration']],
    impact_type_df[['prompt', 'impact_type']],
    on='prompt',
    how='inner'
)

# JSON 형식으로 변환
results = []
for _, row in merged_df.iterrows():
    result_item = {
        "prompt": row["prompt"],
        "impact_duration": row["impact_duration"],
        "impact_type": row["impact_type"]
    }
    results.append(result_item)

output_json_fp = '../results/gpt4-classification/Korean_FIT_1.json'

# 저장
with open(output_json_fp, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
