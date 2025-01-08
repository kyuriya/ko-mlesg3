import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 경로
YI_KO_MODEL = "beomi/Yi-Ko-6B"
EEVE_MODEL = "yanolja/EEVE-Korean-10.8B-v1.0"

# 모델 및 토크나이저 로드
tokenizers = {
    "Yi-Ko-6B": AutoTokenizer.from_pretrained(YI_KO_MODEL),
    "EEVE-Korean-10.8B": AutoTokenizer.from_pretrained(EEVE_MODEL),
}
models = {
    "Yi-Ko-6B": AutoModelForCausalLM.from_pretrained(YI_KO_MODEL),
    "EEVE-Korean-10.8B": AutoModelForCausalLM.from_pretrained(EEVE_MODEL),
}

# 데이터 로드
trainfp = '../data/ML-ESG-3_Trainset_Korean.json'
testfp = '../data/ML-ESG3_Testset_Korean.json'

with open(trainfp, 'r') as f:
    train_data = json.load(f)

with open(testfp, 'r') as f:
    test_data = json.load(f)

# 하이퍼파라미터 설정
shots = [1, 3, 5]
orders = ['standard_order', 'reverse_order']
guidelines = ['msci_simple', 'standard']
tasks = ['Impact Type', 'Impact Duration']
results = []

# In-Context Learning을 통한 예제 생성
def generate_icl_examples(train_data, n_shots, order, task):
    if order == 'reverse_order':
        examples = train_data[-n_shots:]  # Reverse order
    else:
        examples = train_data[:n_shots]  # Standard order

    icl = "\n\n".join([
        f"### text: {example['content']}\n### response: Based on the MSCI guidelines, I classify this to <{example[task.lower().replace(' ', '_')]}>" 
        for example in examples
    ])
    return icl

# Helper function to get model predictions
def get_model_prediction(model_name, tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=512)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# Main calibration 
for model_name in tqdm(models.keys()):
    tokenizer = tokenizers[model_name]
    model = models[model_name]

    for task in tasks:
        for n_shot in shots:
            for order in orders:
                for guideline in guidelines:
                    correct_count = 0
                    total_count = 0
                    correct_logits = []
                    wrong_logits = []

                    for item in test_data:
                        # In-Context Learning을 통한 예제 추가
                        icl_examples = generate_icl_examples(train_data, n_shot, order, task)
                        prompt = f"""
                        You will be given a text. Refer to the examples for your decision. 
                        Classify it to either [cannot distinguish/risk/opportunity] or [short/medium/long] 
                        based on the task and the guidelines below.\n\n
                        {icl_examples}\n\n### text: {item['content']}\n### response:
                        """

                        # 모델 예측
                        prediction = get_model_prediction(model_name, tokenizer, model, prompt)
                        predicted_label = prediction.split("<")[-1].split(">")[0]

                        # Ground truth
                        ground_truth = item[task.lower().replace(" ", "_")]

                        # 정확도와 logits 계산
                        confidence = np.random.uniform(0.5, 0.95)  # Placeholder for confidence
                        if predicted_label == ground_truth:
                            correct_count += 1
                            correct_logits.append(confidence)
                        else:
                            wrong_logits.append(confidence)
                        total_count += 1

                    # 정확도와 평균 logits 계산
                    accuracy = correct_count / total_count
                    avg_correct_logits = np.mean(correct_logits) if correct_logits else 0
                    avg_wrong_logits = np.mean(wrong_logits) if wrong_logits else 0

                    # 결과 저장
                    results.append({
                        'type': f'{n_shot}-shot-{order}-{guideline}',
                        'accuracy': accuracy,
                        'correct_logits': avg_correct_logits,
                        'wrong_logits': avg_wrong_logits,
                        'model': model_name,
                        'task': task
                    })

# 결과 데이터프레임 생성 및 저장
df = pd.DataFrame(results)
df.to_csv('calibration.csv', index=False)