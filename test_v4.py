from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
from PIL import Image
import json

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

# 0~1 정규화 함수
def clip01(v): return max(0.0, min(1.0, float(v)))

def clip_bbox(b):
    x, y, w, h = map(float, b)
    x, y, w, h = clip01(x), clip01(y), clip01(w), clip01(h)
    if x + w > 1: w = 1 - x
    if y + h > 1: h = 1 - y
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]

# fallback layout 추가 (if missing)
def inject_fallback(parsed):
    layout = parsed.get("layout", {})
    subj = layout.get("subject_layout", {"center": [0.5, 0.5], "ratio": [0.3, 0.3]})
    cx, cy = subj.get("center", [0.5, 0.5])
    rw, rh = subj.get("ratio", [0.3, 0.3])
    margin = 0.04

    # 텍스트 없으면 상/하 배너 삽입
    if not layout.get("nongraphic_layout"):
        layout["nongraphic_layout"] = [
            {"type": "headline", "bbox": clip_bbox([margin, margin, 1 - 2*margin, 0.12])},
            {"type": "headline", "bbox": clip_bbox([margin, 1 - margin - 0.12, 1 - 2*margin, 0.12])}
        ]

    # 로고 없으면 우상단 삽입
    if not layout.get("graphic_layout"):
        logo_w, logo_h = 0.25, 0.10
        layout["graphic_layout"] = [
            {"type": "logo", "content": "", "bbox": clip_bbox([1 - margin - logo_w, margin, logo_w, logo_h])}
        ]

    parsed["layout"] = layout
    return parsed

# 이미지 인코딩
base64_image = encode_image_to_base64(image_path)

# GPT 요청
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    messages=[
        {
            "role": "system",
            "content": (
                "오직 다음 JSON만 출력해. 마크다운, 설명 없이 JSON만 반환.\n"
                "형식: {\n"
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
    ]
)

# JSON 응답 파싱
try:
    raw_json = response.choices[0].message.content
    json_start = raw_json.find("{")
    json_end = raw_json.rfind("}") + 1
    parsed = json.loads(raw_json[json_start:json_end])
except Exception as e:
    print("[⚠️ 오류] GPT 응답 파싱 실패:", e)
    print(raw_json)
    exit(1)

# 좌표 정규화
try:
    W, H = Image.open(image_path).size
except:
    W, H = 1, 1

# subject_layout 정규화
if "layout" in parsed and "subject_layout" in parsed["layout"]:
    c = parsed["layout"]["subject_layout"].get("center", [0.5, 0.5])
    r = parsed["layout"]["subject_layout"].get("ratio", [0.3, 0.3])
    if max(c) > 1.0 or max(r) > 1.0:
        c = [clip01(float(c[0]) / W), clip01(float(c[1]) / H)]
        r = [clip01(float(r[0]) / W), clip01(float(r[1]) / H)]
    parsed["layout"]["subject_layout"] = {"center": c, "ratio": r}

# bbox 정규화
for section in ["nongraphic_layout", "graphic_layout"]:
    items = parsed["layout"].get(section, [])
    for item in items:
        if "bbox" in item:
            b = item["bbox"]
            if max(b) > 1.0:
                item["bbox"] = clip_bbox([b[0]/W, b[1]/H, b[2]/W, b[3]/H])
            else:
                item["bbox"] = clip_bbox(b)

# fallback 박스 삽입
parsed = inject_fallback(parsed)

# 최종 출력
print(json.dumps(parsed, ensure_ascii=False, indent=2))

