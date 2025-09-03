from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END, START
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import base64
import time

# --- 1. 환경 변수 로드 ---
load_dotenv()
api_key = os.getenv('OPEN_API_KEY')
client = OpenAI(api_key=api_key)

# --- 2. 상태(State) 정의 ---
class AdGenerationState(TypedDict):
    """LangGraph의 상태를 정의하는 TypedDict"""
    product_name: str
    image_base64: str
    features: Dict[str, Any]
    trends: Dict[str, Any]
    copy: Dict[str, Any]
    background: Dict[str, Any]
    graphic_elements: List[Dict[str, Any]]
    layouts: Dict[str, List[Dict[str, Any]]]
    final_json: List[Dict[str, Any]]

# --- 3. 에이전트 노드 구현 (GPT 호출 로직 통합) ---
class ProductAnalyzerAgent:
    """제품 이미지와 설명을 분석하여 특징, 용도, 마스크 정보를 추출하는 에이전트."""
    def invoke(self, state: AdGenerationState) -> Dict[str, Any]:
        print("➡️ ProductAnalyzerAgent: 제품 이미지 분석 및 특징 추출 중...")
        product_name = state.get("product_name")
        base64_image = state.get("image_base64")
        if not product_name or not base64_image:
            raise ValueError("제품 이름 또는 이미지가 상태에 존재하지 않습니다.")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[{
                    "role": "system",
                    "content": (
                        "You are a professional product designer. Your task is to analyze a product image "
                        "and provide a detailed, objective description. You must analyze the product and its "
                        "ideal background separately. The final output should be a JSON object.\n\n"
                        "The JSON should contain the following keys:\n"
                        "- `product_features`: A detailed description of the product's visual characteristics, texture, and style.\n"
                        "- `use_case`: The primary use or purpose of the product.\n"
                        "- `product_mask`: A technical description of how to create a product mask. Describe it as if providing instructions to an image-editing AI.\n\n"
                        "Please ensure your analysis is based solely on the product's aesthetics, not the background of the input image. "
                        "The output must be a valid JSON object only."
                    )
                }, {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"제품 이름: {product_name}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            response_content = response.choices[0].message.content
            parsed_json = json.loads(response_content)
            result = {
                "features": {
                    "product_features": parsed_json.get("product_features", ""),
                    "use_case": parsed_json.get("use_case", ""),
                    "product_mask": parsed_json.get("product_mask", "")
                }
            }
            print("✅ ProductAnalyzerAgent: 분석 완료")
            print(f"🔍 ProductAnalyzerAgent 결과: {json.dumps(result, ensure_ascii=False, indent=2)}\n")
            return result
        except Exception as e:
            print(f"❌ ProductAnalyzerAgent 오류 발생: {e}")
            return {"features": {}}


class TrendInsightAgent:
    """제품 카테고리의 마케팅 트렌드를 분석하는 에이전트."""
    def invoke(self, state: AdGenerationState) -> Dict[str, Any]:
        print("➡️ TrendInsightAgent: 마케팅 트렌드 분석 중...")
        product_name = state.get("product_name")
        if not product_name:
            raise ValueError("제품 이름이 상태에 존재하지 않습니다.")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[{
                    "role": "system",
                    "content": (
                        "You are a marketing trend analyst. Analyze the product category for the given product. "
                        "The output must be a JSON object containing the following keys:\n"
                        "- `category`: The main product category.\n"
                        "- `popular_brands`: A list of popular brands in this category.\n"
                        "- `slogans`: A list of 3-4 example slogans for this category.\n"
                        "- `tone`: The dominant marketing tone (e.g., 'elegant', 'energetic', 'minimalist')."
                    )
                }, {
                    "role": "user",
                    "content": f"제품 이름: {product_name}"
                }],
                response_format={"type": "json_object"}
            )
            parsed_json = json.loads(response.choices[0].message.content)
            result = {"trends": parsed_json}
            print("✅ TrendInsightAgent: 분석 완료")
            print(f"🔍 TrendInsightAgent 결과: {json.dumps(result, ensure_ascii=False, indent=2)}\n")
            return result
        except Exception as e:
            print(f"❌ TrendInsightAgent 오류 발생: {e}")
            return {"trends": {}}


class MarketingCopyAgent:
    """트렌드와 제품 정보를 기반으로 광고 문구를 생성하는 에이전트."""
    def invoke(self, state: AdGenerationState) -> Dict[str, Any]:
        print("➡️ MarketingCopyAgent: 광고 문구 생성 중...")
        product_features = state.get("features", {}).get("product_features", "")
        trends = state.get("trends", {})
        if not product_features or not trends:
            raise ValueError("제품 특징 또는 트렌드 정보가 상태에 존재하지 않습니다.")
        try:
            prompt = f"제품 특징: {product_features}\n마케팅 트렌드: {trends}\n\n위 정보를 바탕으로 다음의 JSON을 생성해주세요:\n- `logo`: 제품에 어울리는 로고 텍스트 (최대 1단어)\n- `tagline`: 강력한 광고 태그라인\n- `underlay`: 제품을 보조하는 짧은 문구"
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a creative copywriter. Your output must be a valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            parsed_json = json.loads(response.choices[0].message.content)
            result = {"copy": parsed_json}
            print("✅ MarketingCopyAgent: 생성 완료")
            print(f"🔍 MarketingCopyAgent 결과: {json.dumps(result, ensure_ascii=False, indent=2)}\n")
            return result
        except Exception as e:
            print(f"❌ MarketingCopyAgent 오류 발생: {e}")
            return {"copy": {}}


class BackgroundDesignerAgent:
    """이상적인 광고 배경을 설명하고 이미지 생성 프롬프트를 만드는 에이전트."""
    def invoke(self, state: AdGenerationState) -> Dict[str, Any]:
        print("➡️ BackgroundDesignerAgent: 배경 설명 및 프롬프트 생성 중...")
        product_features = state.get("features", {}).get("product_features", "")
        if not product_features:
            raise ValueError("제품 특징 정보가 상태에 존재하지 않습니다.")
        try:
            prompt = f"제품 특징: {product_features}\n\n이 제품을 가장 잘 돋보이게 할 광고 배경에 대해 JSON을 생성해주세요.\n- `background_caption`: 배경에 대한 설명 (1-2문장)\n- `background_prompt`: AI 이미지 생성용 프롬프트"
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a professional set designer for product photography. Your output must be a valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            parsed_json = json.loads(response.choices[0].message.content)
            result = {"background": parsed_json}
            print("✅ BackgroundDesignerAgent: 생성 완료")
            print(f"🔍 BackgroundDesignerAgent 결과: {json.dumps(result, ensure_ascii=False, indent=2)}\n")
            return result
        except Exception as e:
            print(f"❌ BackgroundDesignerAgent 오류 발생: {e}")
            return {"background": {}}


class GraphicElementAgent:
    """광고 문구와 제품 특징을 기반으로 그래픽 요소의 타입과 내용을 결정하는 에이전트."""
    def invoke(self, state: AdGenerationState) -> Dict[str, Any]:
        print("➡️ GraphicElementAgent: 그래픽 요소 내용 결정 중...")
        copy = state.get("copy", {})
        if not copy:
            raise ValueError("광고 문구(copy)가 상태에 존재하지 않습니다.")
        
        graphic_elements = [
            {"type": "tagline", "content": copy.get("tagline", "")},
            {"type": "underlay", "content": copy.get("underlay", "")},
            {"type": "logo", "content": copy.get("logo", "")}
        ]
        
        result = {"graphic_elements": graphic_elements}
        print("✅ GraphicElementAgent: 결정 완료")
        print(f"🔍 GraphicElementAgent 결과: {json.dumps(result, ensure_ascii=False, indent=2)}\n")
        return result


class AspectRatioPlannerAgent:
    """4가지 종횡비에 맞는 요소 배치(Bounding Box)를 설계하는 에이전트."""
    def invoke(self, state: AdGenerationState) -> Dict[str, Any]:
        print("➡️ AspectRatioPlannerAgent: 종횡비별 레이아웃 설계 중...")
        graphic_elements = state.get("graphic_elements")
        product_features = state.get("features", {}).get("product_features")
        
        if not graphic_elements or not product_features:
            raise ValueError("그래픽 요소 또는 제품 특징 정보가 상태에 존재하지 않습니다.")
            
        prompt = f"""
I will provide you with a product's description, its ideal foreground, background prompt, and several taglines. Please design a beautiful layout for a poster.

- **Required advertising size**: 800x1200
- **Task description**: Design a poster layout based on the provided information.
- **Product features**: {product_features}
- **Graphic elements**: {json.dumps(graphic_elements, ensure_ascii=False)}

The output must be a JSON object with the following structure for each aspect ratio (0.684, 1.0, 0.667, 0.75):
- `target canvas aspect ratio`: The aspect ratio of the canvas.
- `foreground prompt`: The prompt for the main subject.
- `background prompt`: The prompt for the background.
- `subject layout`: A list of layouts for the main subject, including its type, bbox, and aspect ratio.
- `nongraphic layout`: A list of layouts for any non-graphic elements like tables or shapes.
- `graphic layout`: A list of layouts for graphic elements like taglines and logos, including their type, content, and bbox.

The bounding box (`bbox`) must be an array of four values [x1, y1, x2, y2] relative to the canvas size (e.g., [0.495, 0.644, 0.493, 0.493]). Ensure the design is aesthetically pleasing for each aspect ratio.
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a professional graphic designer. Your output must be a valid JSON object only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            parsed_json = json.loads(response.choices[0].message.content)
            result = {"layouts": parsed_json}
            print("✅ AspectRatioPlannerAgent: 설계 완료")
            print(f"🔍 AspectRatioPlannerAgent 결과: {json.dumps(result, ensure_ascii=False, indent=2)}\n")
            return result
        except Exception as e:
            print(f"❌ AspectRatioPlannerAgent 오류 발생: {e}")
            return {"layouts": {}}


class SceneAssemblerAgent:
    """모든 결과를 통합해 최종 JSON을 생성하는 에이전트."""
    def invoke(self, state: AdGenerationState) -> Dict[str, Any]:
        print("➡️ SceneAssemblerAgent: 최종 JSON 조립 중...")
        final_scenes = []
        aspect_ratios = ["0.684", "1.0", "0.667", "0.75"]
        
        required_keys = ["features", "background", "copy", "layouts", "graphic_elements"]
        if not all(key in state and state.get(key) for key in required_keys):
            raise ValueError("필수 데이터가 상태에 존재하지 않습니다.")

        for ratio in aspect_ratios:
            layout_data = state["layouts"].get(ratio, {})
            if not layout_data:
                continue

            scene = {
                "aspect_ratio": float(ratio),
                "foreground_caption": layout_data.get("foreground prompt", state["features"]["product_features"]),
                "background_caption": layout_data.get("background prompt", state["background"]["background_caption"]),
                "ai_background_prompt": state["background"]["background_prompt"],
                "layout": {
                    "subject": layout_data.get("subject layout", []),
                    "nongraphic": layout_data.get("nongraphic layout", []),
                    "graphic": layout_data.get("graphic layout", [])
                },
                "product_mask": state["features"]["product_mask"],
            }
            final_scenes.append(scene)
        result = {"final_json": final_scenes}
        print("✅ SceneAssemblerAgent: 조립 완료")
        print(f"🔍 SceneAssemblerAgent 결과: {json.dumps(result, ensure_ascii=False, indent=2)}\n")
        return result



# --- 4. LangGraph DAG 구성 (수정) ---
def create_graph():
    """LangGraph를 생성하고 노드와 엣지를 연결"""
    graph_builder = StateGraph(AdGenerationState)

    # 노드 추가
    graph_builder.add_node("product_analyzer", ProductAnalyzerAgent().invoke)
    graph_builder.add_node("trend_insight", TrendInsightAgent().invoke)
    graph_builder.add_node("marketing_copy", MarketingCopyAgent().invoke)
    graph_builder.add_node("background_designer", BackgroundDesignerAgent().invoke)
    graph_builder.add_node("graphic_element", GraphicElementAgent().invoke)
    graph_builder.add_node("aspect_ratio_planner", AspectRatioPlannerAgent().invoke)
    graph_builder.add_node("scene_assembler", SceneAssemblerAgent().invoke)

    # DAG 엣지 연결
    # Step 1: 병렬 시작
    graph_builder.add_edge(START, "product_analyzer")
    graph_builder.add_edge(START, "trend_insight")

    # Step 2: marketing_copy는 두 분석 결과 모두 기다림
    graph_builder.add_edge("product_analyzer", "marketing_copy")
    graph_builder.add_edge("trend_insight", "marketing_copy")

    # Step 3: background_designer는 product_analyzer만 필요
    graph_builder.add_edge("product_analyzer", "background_designer")
    graph_builder.add_edge("product_analyzer", "aspect_ratio_planner")

    # Step 4: marketing_copy 결과로 graphic 요소 생성
    graph_builder.add_edge("marketing_copy", "graphic_element")
    graph_builder.add_edge("product_analyzer", "graphic_element")

    # Step 5: graphic 요소 → aspect ratio layout
    graph_builder.add_edge("graphic_element", "aspect_ratio_planner")
    graph_builder.add_edge("aspect_ratio_planner", "scene_assembler")

    # Step 6: background + layout → scene 조립
    graph_builder.add_edge("background_designer", "scene_assembler")
    

    # 마지막 완료
    graph_builder.add_edge("scene_assembler", END)


    return graph_builder.compile()

# --- 5. 실행 로직 ---
def encode_image_to_base64(image_path):
    """이미지 파일을 base64로 인코딩하는 헬퍼 함수."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"오류: '{image_path}' 파일을 찾을 수 없습니다. 올바른 경로를 입력해주세요.")
        return None

if __name__ == "__main__":
    print("✨ 멀티모달 광고 생성 시스템 시작 ✨")
    
    product_name = input("제품 이름을 입력하세요: ")
    image_path = input("제품 이미지 파일 경로를 입력하세요 (예: './image.jpg'): ")
    
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        exit()

    initial_state = {
        "product_name": product_name,
        "image_base64": base64_image
    }

    graph = create_graph()
    
    print("\n🚀 광고 생성 워크플로우를 시작합니다...")
    start_time = time.time()
    
    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        print(f"워크플로우 실행 중 치명적인 오류 발생: {e}")
        final_state = {"final_json": []} # 오류 발생 시 빈 JSON 반환

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n✅ 모든 에이전트 작업 완료. 최종 광고 구성 JSON 출력:")
    print(json.dumps(final_state["final_json"], indent=4, ensure_ascii=False))
    print(f"\n총 소요 시간: {elapsed_time:.2f}초")