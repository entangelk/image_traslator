import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
from models.text_block import TextBlock, TextStyle


class ImageProcessor:
    def __init__(self, default_font_path: Optional[str] = None):
        self.default_font_path = default_font_path
    
    def remove_text_regions(self, image_path: str, text_blocks: List[TextBlock]) -> np.ndarray:
        image = cv2.imread(image_path)
        
        for block in text_blocks:
            # 텍스트 영역을 마스크로 처리
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (block.x, block.y), 
                         (block.x + block.width, block.y + block.height), 255, -1)
            
            # 인페인팅을 사용해 텍스트 영역 복원
            image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        return image
    
    def insert_translated_text(self, image: np.ndarray, text_blocks: List[TextBlock]) -> np.ndarray:
        # OpenCV 이미지를 PIL로 변환
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        for block in text_blocks:
            if not block.translated_text:
                continue
                
            # 폰트 설정
            font = self._get_font(block.style)
            
            # 텍스트 색상 설정
            color = self._get_text_color(block.style)
            
            # 텍스트 위치 계산 (중앙 정렬)
            text_bbox = draw.textbbox((0, 0), block.translated_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = block.x + (block.width - text_width) // 2
            y = block.y + (block.height - text_height) // 2
            
            # 배경색이 있다면 배경 그리기
            if block.style and block.style.background_color:
                bg_color = block.style.background_color
                draw.rectangle([x-2, y-2, x+text_width+2, y+text_height+2], fill=bg_color)
            
            # 텍스트 그리기
            draw.text((x, y), block.translated_text, font=font, fill=color)
        
        # PIL 이미지를 다시 OpenCV로 변환
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _get_font(self, style: Optional[TextStyle]) -> ImageFont.FreeTypeFont:
        if style and style.font_size:
            font_size = style.font_size
        else:
            font_size = 20
            
        if self.default_font_path:
            try:
                return ImageFont.truetype(self.default_font_path, font_size)
            except:
                pass
        
        try:
            return ImageFont.truetype("arial.ttf", font_size)
        except:
            return ImageFont.load_default()
    
    def _get_text_color(self, style: Optional[TextStyle]) -> Tuple[int, int, int]:
        if style and style.color:
            return style.color
        return (0, 0, 0)  # 기본 검은색
    
    def estimate_text_style(self, image: np.ndarray, text_block: TextBlock) -> TextStyle:
        # 텍스트 영역의 색상 분석
        x, y, w, h = text_block.x, text_block.y, text_block.width, text_block.height
        roi = image[y:y+h, x:x+w]
        
        # 가장 어두운 색을 텍스트 색으로 추정
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text_color_val = int(np.percentile(gray_roi, 10))  # 하위 10%
        
        # 가장 밝은 색을 배경색으로 추정
        bg_color_val = int(np.percentile(gray_roi, 90))    # 상위 10%
        
        # 폰트 크기 추정 (높이 기반)
        estimated_font_size = max(12, min(72, h - 4))
        
        return TextStyle(
            font_size=estimated_font_size,
            color=(text_color_val, text_color_val, text_color_val),
            background_color=(bg_color_val, bg_color_val, bg_color_val) if bg_color_val - text_color_val > 50 else None
        )