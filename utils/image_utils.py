import cv2
import numpy as np
from typing import Tuple, Optional


def resize_image(image: np.ndarray, max_width: int = 1920, max_height: int = 1080) -> np.ndarray:
    """이미지 크기를 최대 크기에 맞게 조정"""
    height, width = image.shape[:2]
    
    if width <= max_width and height <= max_height:
        return image
    
    # 비율 유지하면서 크기 조정
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def enhance_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """OCR 성능 향상을 위한 이미지 전처리"""
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 노이즈 제거
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 샤프닝
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened


def validate_image(image_path: str) -> bool:
    """이미지 파일 유효성 검사"""
    try:
        image = cv2.imread(image_path)
        return image is not None and image.size > 0
    except:
        return False


def get_image_info(image_path: str) -> Optional[dict]:
    """이미지 정보 추출"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'size': image.size,
            'dtype': str(image.dtype)
        }
    except:
        return None


def create_text_mask(image: np.ndarray, text_blocks) -> np.ndarray:
    """텍스트 영역에 대한 마스크 생성"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for block in text_blocks:
        cv2.rectangle(mask, 
                     (block.x, block.y), 
                     (block.x + block.width, block.y + block.height), 
                     255, -1)
    
    return mask