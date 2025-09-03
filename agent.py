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

#ProductAnalyzerAgent    
def product_analyzer_agent(image_path, product_name):
    base64_image = encode_image_to_base64(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "제품 분석... (features, use_case, mask) 생성"},
            {"role": "user", "content": [
                {"type": "text", "text": f"제품 이름: {product_name}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
    )
    return response.choices[0].message.content

#BackgroundDesignerAgent
def background_designer_agent(product_features_json):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "배경 설명 및 이미지 생성 프롬프트 생성"},
            {"role": "user", "content": product_features_json}
        ]
    )
    return response.choices[0].message.content


#TrendInsightAgent
def trend_insight_agent(product_description):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "마케팅 트렌드 조사 및 문구 예시 수집"},
            {"role": "user", "content": f"Product description: {product_description}"}
        ]
    )
    return response.choices[0].message.content


#MarketingCopyAgent
def marketing_copy_agent(product_name, image_path, trend_insight):
    base64_image = encode_image_to_base64(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Logo, Tagline, Underlay 생성 (트렌드 반영)"},
            {"role": "user", "content": [
                {"type": "text", "text": f"제품 이름: {product_name}\n트렌드: {trend_insight}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
    )
    return response.choices[0].message.content


#GraphicElementAgent
def graphic_element_agent(product_analysis, marketing_copy_json):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "그래픽 요소 유형 및 위치 계획 (bbox 포함)"},
            {"role": "user", "content": f"분석: {product_analysis}\n마케팅 문구: {marketing_copy_json}"}
        ]
    )
    return response.choices[0].message.content


#AspectRatioPlannerAgent
def layout_planner_agent(product_analysis, aspect_ratios=[0.684, 1.0, 0.667, 0.75]):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "종횡비별 요소 배치 계획"},
            {"role": "user", "content": f"{product_analysis}"}
        ]
    )
    return response.choices[0].message.content


#SceneAssemblerAgent
def scene_assembler_agent(foreground_caption, background_caption, background_prompt,
                          layouts, product_mask, graphic_elements):
    return {
        "aspect_ratios": [0.684, 1.0, 0.667, 0.75],
        "scenes": [
            {
                "aspect_ratio": ratio,
                "foreground_caption": foreground_caption,
                "background_caption": background_caption,
                "background_prompt": background_prompt,
                "product_mask": product_mask,
                "layout": layouts.get(str(ratio), []),
                "graphic_elements": graphic_elements
            }
            for ratio in [0.684, 1.0, 0.667, 0.75]
        ]
    }


features_json = product_analyzer_agent(image_path, product_name)
trend_json = trend_insight_agent(features_json)
marketing_json = marketing_copy_agent(product_name, image_path, trend_json)
background_json = background_designer_agent(features_json)
layout_json = layout_planner_agent(features_json)
graphic_json = graphic_element_agent(features_json, marketing_json)

import json
final_output = scene_assembler_agent(
    foreground_caption=product_name,
    background_caption=json.loads(background_json)["background_caption"],
    background_prompt=json.loads(background_json)["background_prompt"],
    layouts=json.loads(layout_json)["layouts"],
    product_mask=json.loads(features_json)["product_mask"],
    graphic_elements=json.loads(marketing_json)
)

print(json.dumps(final_output, indent=2, ensure_ascii=False))
