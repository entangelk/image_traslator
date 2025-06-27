import cv2
import os
from typing import List, Optional
from core.ocr_detector import OCRDetector
from core.translator import TextTranslator
from core.image_processor import ImageProcessor
from core.style_analyzer import StyleAnalyzer
from models.text_block import TextBlock


class ImageTranslator:
    def __init__(self, 
                 source_lang='auto', 
                 target_lang='ko',
                 translation_engine='google',
                 ocr_engine='easyocr',
                 font_path: Optional[str] = None):
        
        self.ocr_detector = OCRDetector(lang='multilingual', engine=ocr_engine)
        self.translator = TextTranslator(source_lang, target_lang, translation_engine)
        self.image_processor = ImageProcessor(font_path)
        self.style_analyzer = StyleAnalyzer()
    
    def translate_image(self, 
                       input_path: str, 
                       output_path: str,
                       confidence_threshold: float = 0.5) -> bool:
        
        if not os.path.exists(input_path):
            print(f"Error: Input file {input_path} not found")
            return False
        
        try:
            # 1. 텍스트 감지
            print("Detecting text...")
            text_blocks = self.ocr_detector.detect_text(input_path, confidence_threshold)
            
            if not text_blocks:
                print("No text detected in the image")
                return False
            
            print(f"Found {len(text_blocks)} text blocks")
            
            # 2. 스타일 분석
            print("Analyzing text styles...")
            original_image = cv2.imread(input_path)
            for block in text_blocks:
                block.style = self.style_analyzer.analyze_text_style(original_image, block)
            
            # 3. 번역
            print("Translating text...")
            translated_blocks = self.translator.translate_blocks(text_blocks)
            
            # 4. 이미지 처리
            print("Processing image...")
            # 원본 텍스트 제거
            processed_image = self.image_processor.remove_text_regions(input_path, translated_blocks)
            
            # 번역된 텍스트 삽입
            final_image = self.image_processor.insert_translated_text(processed_image, translated_blocks)
            
            # 5. 결과 저장
            cv2.imwrite(output_path, final_image)
            print(f"Translation completed. Output saved to {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error during translation: {e}")
            return False
    
    def preview_detected_text(self, input_path: str) -> List[TextBlock]:
        """디버깅용: 감지된 텍스트 블록들을 반환"""
        return self.ocr_detector.detect_text(input_path)
    
    def set_languages(self, source_lang: str, target_lang: str):
        """언어 설정 변경"""
        self.translator = TextTranslator(source_lang, target_lang, self.translator.engine)
        
        # OCR은 다국어 감지로 고정 (언어 변경과 무관)
        # self.ocr_detector = OCRDetector(lang='multilingual')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Translate text in images')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--source-lang', default='auto', help='Source language (default: auto)')
    parser.add_argument('--target-lang', default='ko', help='Target language (default: ko)')
    parser.add_argument('--confidence', type=float, default=0.5, help='OCR confidence threshold')
    parser.add_argument('--font-path', help='Path to font file for rendering')
    parser.add_argument('--ocr-engine', default='easyocr', choices=['easyocr', 'paddleocr'], help='OCR engine choice')
    parser.add_argument('--preview', action='store_true', help='Preview detected text without translation')
    
    args = parser.parse_args()
    
    translator = ImageTranslator(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        ocr_engine=args.ocr_engine,
        font_path=args.font_path
    )
    
    if args.preview:
        # 미리보기 모드
        text_blocks = translator.preview_detected_text(args.input)
        print(f"Detected {len(text_blocks)} text blocks:")
        for i, block in enumerate(text_blocks):
            print(f"{i+1}. '{block.original_text}' at ({block.x}, {block.y}) "
                  f"size: {block.width}x{block.height}, confidence: {block.confidence:.2f}")
    else:
        # 번역 실행
        success = translator.translate_image(args.input, args.output, args.confidence)
        if not success:
            exit(1)


if __name__ == "__main__":
    main()