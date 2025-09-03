from openai import OpenAI
from dotenv import load_dotenv
import os
from PIL import Image
import base64

# Load API Key
load_dotenv()
api_key = os.getenv('OPEN_API_KEY')
client = OpenAI(api_key=api_key)

# 사용자 입력 받기
product_name = input("제품 이름을 입력하세요: ")
image_path = input("제품 이미지 파일 경로를 입력하세요 (예: './image.jpg'): ")

# 이미지 파일을 base64로 인코딩
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 이미지 인코딩
base64_image = encode_image_to_base64(image_path)

# GPT 요청
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.7,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a professional designer. What you see is a product. You need to follow the instructions below:\n\n"

                "# <front description>\n"
                "Please give a description of the product.\n\n"

                "# <back description>\n"
                "Please use professional designer aesthetics to describe the background in which the product looks best. Please give a detailed and objective description. The background should:"
                "- Highlight the product's features and use-case."
                "- Include details such as location, lighting, mood, color scheme, and materials."
                "- Avoid subjective or vague terms."
                "- Be visually neutral enough to not overpower the product.\n\n"

                "# Additionally:\n"
                "Generate a concise background prompt suitable for use with AI image generation models like Stable Diffusion. The prompt should include key visual elements such as:"
                "- Location/setting"
                "- Lighting"
                "- Mood"
                "- Color palette"
                "- Background elements or materials\n\n"

                "# Finally\n"
                "Please output your predicted value in JSON format. Analyze the product and the background *separately*, and do not assume the current background is optimal. Recommend the best possible background based on product aesthetics, not what you see in the image."

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
