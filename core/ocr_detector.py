import cv2
import numpy as np
from typing import List, Optional
from paddleocr import PaddleOCR
import easyocr
from models.text_block import TextBlock, TextStyle


class OCRDetector:
    def __init__(self, use_angle_cls=True, lang='multilingual', engine='easyocr'):
        self.lang = lang
        self.engine = engine
        
        if engine == 'easyocr' and lang == 'multilingual':
            # EasyOCR로 80개 언어 동시 지원
            self.ocr = easyocr.Reader(['ko', 'en', 'ja', 'zh-cn', 'zh-tw', 'th', 'vi', 'ar'])
        elif engine == 'paddleocr':
            if lang == 'multilingual':
                # PaddleOCR 중국어 모델 (영어+중국어+숫자)
                self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang='ch')
            else:
                self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
        else:
            # 기본값: EasyOCR 다국어
            self.ocr = easyocr.Reader(['ko', 'en', 'ja', 'zh-cn', 'zh-tw'])
    
    def detect_text(self, image_path: str, confidence_threshold: float = 0.5) -> List[TextBlock]:
        text_blocks = []
        
        if self.engine == 'easyocr':
            # EasyOCR 결과 처리
            results = self.ocr.readtext(image_path)
            
            for result in results:
                if len(result) < 3:
                    continue
                    
                bbox, text, confidence = result
                
                if confidence < confidence_threshold:
                    continue
                
                # EasyOCR bbox 형식: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x = int(min(x_coords))
                y = int(min(y_coords))
                width = int(max(x_coords) - min(x_coords))
                height = int(max(y_coords) - min(y_coords))
                
                text_block = TextBlock(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    original_text=text,
                    confidence=confidence
                )
                
                text_blocks.append(text_block)
                
        else:
            # PaddleOCR 결과 처리
            results = self.ocr.ocr(image_path, cls=True)
            
            if not results or not results[0]:
                return text_blocks
                
            for line in results[0]:
                if len(line) < 2:
                    continue
                    
                bbox, (text, confidence) = line
                
                if confidence < confidence_threshold:
                    continue
                    
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x = int(min(x_coords))
                y = int(min(y_coords))
                width = int(max(x_coords) - min(x_coords))
                height = int(max(y_coords) - min(y_coords))
                
                text_block = TextBlock(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    original_text=text,
                    confidence=confidence
                )
                
                text_blocks.append(text_block)
        
        return text_blocks
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced