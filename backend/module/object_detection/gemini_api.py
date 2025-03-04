from google import genai
from google.genai import types

class GeminiAPI:
    """
    A class for interacting with Google's Gemini API to process images and return bounding box data.

    Attributes:
        api_key (str): The API key for authentication with Google Gemini API.
        model_name (str): The specific Gemini model being used.
        prompt (str): The text prompt guiding the model's response.
        img (any): The image input to be processed by the model.
        safety_settings (list): A set of safety rules to filter harmful content.
        bounding_box_system_instructions (str): Instructions for the API to return bounding boxes in JSON format.
    """

    def __init__(self, api_key: str, model_name: str, prompt: str, img: any):
        """
        Initializes the GeminiAPI object with necessary parameters.

        Args:
            api_key (str): The API key for accessing Google Gemini API.
            model_name (str): The name of the AI model to use.
            prompt (str): A textual prompt that guides the AI's response.
            img (any): The image input to be analyzed.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.prompt = prompt
        self.img = img
        
        # Define safety settings to block potentially harmful content.
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]

        # Instruction to force API response in structured JSON format with bounding boxes.
        self.bounding_box_system_instructions = """
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
        If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
        """

    def call_api(self):
        """
        Sends a request to the Gemini API with the given image and prompt.

        The method:
        1. Creates an authenticated API client.
        2. Sends the image and prompt data to the model.
        3. Requests a structured JSON response containing labeled bounding boxes.

        Returns:
            API response containing detected objects and bounding boxes in JSON format.
        """
        try:
            # Initialize API client using provided API key
            client = genai.Client(api_key=self.api_key)

            # Send request to the model and retrieve response
            response = client.models.generate_content(
                model=self.model_name,
                contents=[self.prompt, self.img],
                config=types.GenerateContentConfig(
                    system_instruction=self.bounding_box_system_instructions,
                    temperature=0,  # Keep responses consistent (less randomness)
                    safety_settings=self.safety_settings,
                )
            )

            return response  # Return the structured JSON response

        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return None  # Return None in case of failure
if __name__ == "__main__":
    import cv2
    import numpy as np
    import json
    import re
    from PIL import Image
    from io import BytesIO

    # ✅ Hàm trích xuất JSON từ phản hồi Markdown của Gemini
    def extract_json_from_response(response_text):
        """Trích xuất JSON từ phản hồi Markdown của Gemini."""
        try:
            # Loại bỏ phần mô tả "Here are the bounding box detections:"
            response_text = response_text.split("json", 1)[-1].strip()

            # Dùng regex để tìm JSON bên trong
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                json_data = json_match.group(0)  # Lấy phần JSON thuần túy
                return json.loads(json_data)  # Chuyển thành danh sách Python
            else:
                print("❌ Không tìm thấy JSON hợp lệ trong phản hồi.")
                return []
        except json.JSONDecodeError as e:
            print("❌ Lỗi giải mã JSON:", e)
            return []

    # ✅ Hàm vẽ bounding boxes với OpenCV
    def plot_bounding_boxes_opencv(image, bounding_boxes):
        """Vẽ bounding boxes lên ảnh bằng OpenCV."""
        if not isinstance(bounding_boxes, list):
            print("❌ Lỗi: bounding_boxes phải là danh sách.")
            return

        img = np.array(image)  # Convert PIL Image sang NumPy array
        height, width, _ = img.shape
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # BGR Colors

        for i, box in enumerate(bounding_boxes):
            if not isinstance(box, dict) or "box_2d" not in box:
                print("❌ Lỗi: Bounding box không hợp lệ:", box)
                continue  # Bỏ qua box không đúng định dạng

            color = colors[i % len(colors)]  # Chọn màu từ danh sách
            y1 = int(box["box_2d"][0] / 1000 * height)
            x1 = int(box["box_2d"][1] / 1000 * width)
            y2 = int(box["box_2d"][2] / 1000 * height)
            x2 = int(box["box_2d"][3] / 1000 * width)

            # Đảm bảo tọa độ hợp lệ
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # Vẽ khung chữ nhật
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Thêm nhãn nếu có
            if "label" in box:
                cv2.putText(img, box["label"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Hiển thị ảnh với bounding boxes
        cv2.imshow("Bounding Boxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ✅ Đọc ảnh từ file
    image_path = r"backend\frames\test1_20250227_095601\frame_0_01_16.jpg"
    im = Image.open(BytesIO(open(image_path, "rb").read()))
    im.thumbnail([620, 620], Image.Resampling.LANCZOS)

    # ✅ Gọi API Gemini để lấy bounding boxes
    api_key = "AIzaSyA8mDUYiAPLEVrKWYGk3dYG36APdC8MGMc"
    model_name = 'gemini-2.0-flash'
    prompt = """
    Detect objects in the image. Return the output as a JSON list where each entry contains:
    - "box_2d": The 2D bounding box as [y_min, x_min, y_max, x_max].
    - "label": The name of the object (e.g., "bus", "car", "person").
    - "position": The relative position of the object in the image ("left", "right", "top", "bottom", "center", "top left", "bottom right", etc.).

    Ensure that:
    - "label" only contains the object's name, not its position.
    - "position" describes the object's placement.

    Do not return Markdown, explanations, or extra text. Only output valid JSON.
    """

    gemini = GeminiAPI(api_key, model_name, prompt, im)
    response = gemini.call_api()

    # ✅ Trích xuất JSON từ phản hồi
    bounding_boxes = extract_json_from_response(response.text)

    # ✅ Kiểm tra dữ liệu
    print("📌 Bounding boxes:", bounding_boxes)
    print("📌 Kiểu dữ liệu của bounding_boxes:", type(bounding_boxes))

    # ✅ Vẽ bounding boxes lên ảnh
    plot_bounding_boxes_opencv(im, bounding_boxes)
