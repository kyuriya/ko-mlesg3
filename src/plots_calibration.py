import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
file_path = '../results/calibration.csv'
data = pd.read_csv(file_path)

# 모델 및 태스크별 필터링
yi_type = data[(data.model == 'Yi-Ko-6B') & (data.task == 'Impact Type')]
yi_duration = data[(data.model == 'Yi-Ko-6B') & (data.task == 'Impact Duration')]

eeve_type = data[(data.model == 'EEVE-Korean-10.8B') & (data.task == 'Impact Type')]
eeve_duration = data[(data.model == 'EEVE-Korean-10.8B') & (data.task == 'Impact Duration')]


plt.style.use('default')
# Scatter Plot 생성
fig, ax = plt.subplots()
ax.scatter(yi_type['correct_logits'], yi_type['accuracy'], marker="o", color='red', label='Yi-Ko-6B - Impact Type')
ax.scatter(yi_duration['correct_logits'], yi_duration['accuracy'], marker="o", color='blue', label='Yi-Ko-6B - Impact Duration')
ax.scatter(eeve_type['correct_logits'], eeve_type['accuracy'], marker="^", color='red', label='EEVE - Impact Type')
ax.scatter(eeve_duration['correct_logits'], eeve_duration['accuracy'], marker="^", color='blue', label='EEVE - Impact Duration')


ax.plot([0.3, 0.9], [0.3, 0.9], 'k--', label='y=x line') # Assuming accuracy and logits are normalized between 0 and 1 for plotting

ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_xlabel('Confidence (%)', fontsize=14)
ax.legend()
plt.grid()

# Regression Line 추가
sns.regplot(x=data['correct_logits'].values, y=data['accuracy'].values, scatter=False, color='purple')

# 그래프 저장
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
plt.savefig('../results/calibration.pdf', bbox_inches='tight', dpi=300)

# Accuracy 분석 결과 저장
output = []
for (model, task), batch in data.groupby(['model', 'task']):
    mean = batch.accuracy.mean()
    max_val = batch.accuracy.max()
    min_val = batch.accuracy.min()
    output.append([task, model, min_val, max_val, mean])

output_df = pd.DataFrame(output, columns=['Task', 'Model', 'Min Accuracy', 'Max Accuracy', 'Mean Accuracy'])
output_df.to_csv('../results/calibration_summary.csv', index=False)