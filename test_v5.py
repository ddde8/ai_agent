import openai
from dotenv import load_dotenv
import os
import base64

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPEN_API_KEY')

# 사용자 입력 받기
product_name = input("제품 이름을 입력하세요: ")
image_path = input("제품 이미지 파일 경로를 입력하세요 (예: './image.jpg'): ")

# 이미지 파일을 base64로 인코딩
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image_to_base64(image_path)

# GPT 요청
response = openai.chat.completions.create(
    model="gpt-4o",
    temperature=0.7,
    messages=[
        {
            "role": "system",
            "content": (
                '오직 다음 JSON만 출력해. 다른 설명/코멘트/마크다운 금지.\n'
                '형식: {\n'
                '  "product": {\n'
                '    "type": "<제품 종류>",\n'
                '    "material": "<재질>",\n'
                '    "design": "<디자인 요약>",\n'
                '    "features": "<브랜드/각인/색상 등 특징>"\n'
                '  },\n'
                '  "background": {\n'
                '    "ideal_color": "<배경 색상>",\n'
                '    "texture": "<배경 질감>",\n'
                '    "lighting": "<조명 스타일>",\n'
                '    "style": "<연출 스타일>"\n'
                '  },\n'
                '  "layout": {\n'
                '    "subject_layout": {"center":[cx,cy],"ratio":[rw,rh]},\n'
                '    "nongraphic_layout":[{"type":"headline","bbox":[x,y,w,h]},...],\n'
                '    "graphic_layout":[{"type":"logo","content":"...","bbox":[x,y,w,h]},...]\n'
                '  }\n'
                '}'
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"제품 이름: {product_name}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    max_tokens=1000
)

# 응답 출력
print(response.choices[0].message.content)
