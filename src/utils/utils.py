from openai import OpenAI

# OpenAI API 클라이언트 초기화
client = OpenAI(api_key="")  # 여기에 OpenAI API 키 입력

def get_answer(prompt, text, model='gpt-4-0125-preview'):
    """
    GPT-4 모델을 호출하여 답변을 생성합니다.

    Args:
        prompt (str): 모델에게 제공할 프롬프트 텍스트
        text (str): 분류할 실제 텍스트
        model (str): 사용할 GPT-4 모델 버전

    Returns:
        tuple: (프롬프트 텍스트, 생성된 응답 텍스트)
    """
    query = prompt.replace('{text}', text)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a skilled expert in the ESG domain."},
            {"role": "user", "content": query}
        ],
        temperature=0.0  # 응답의 일관성을 위해 온도를 0으로 설정
    )
    return query, completion.choices[0].message.content

def parse_between(s, a, b):
    """
    문자열에서 특정 구간 사이의 텍스트를 추출합니다.

    Args:
        s (str): 입력 문자열
        a (str): 시작 문자열
        b (str): 종료 문자열

    Returns:
        str: 시작 문자열과 종료 문자열 사이의 텍스트
    """
    start_index = s.find(a)
    end_index = s.find(b)

    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return ""

    # 시작 문자열 이후의 텍스트만 추출
    start_index += len(a)
    return s[start_index:end_index]