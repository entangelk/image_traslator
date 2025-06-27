# Image Translator

AI 기반 이미지 텍스트 번역 시스템

## 기능

- 이미지에서 텍스트 자동 감지 (PaddleOCR)
- 다국어 번역 지원 (Google Translate, Deep Translator)
- 원본 텍스트 제거 및 번역된 텍스트 삽입
- 텍스트 스타일 분석 및 보존
- 복잡한 포스터/이미지 처리 지원

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 사용법

```bash
python main.py input.jpg output.jpg
```

### 옵션

```bash
python main.py input.jpg output.jpg --source-lang en --target-lang ko --confidence 0.7
```

### 텍스트 미리보기

```bash
python main.py input.jpg output.jpg --preview
```

## 코드 사용 예제

```python
from main import ImageTranslator

# 번역기 초기화
translator = ImageTranslator(
    source_lang='en',
    target_lang='ko',
    font_path='fonts/NanumGothic.ttf'
)

# 이미지 번역
success = translator.translate_image('input.jpg', 'output.jpg')
```

## 프로젝트 구조

```
image_translator/
├── main.py                 # 메인 실행 파일
├── config/
│   └── settings.py         # 설정 파일
├── core/
│   ├── ocr_detector.py     # 텍스트 감지
│   ├── translator.py       # 번역 처리
│   ├── image_processor.py  # 이미지 편집
│   └── style_analyzer.py   # 스타일 분석
├── utils/
│   └── image_utils.py      # 이미지 유틸리티
├── models/
│   └── text_block.py       # 데이터 모델
└── fonts/                  # 폰트 파일
```

## 지원 언어

- 한국어 (ko)
- 영어 (en)
- 기타 언어는 PaddleOCR 지원에 따라