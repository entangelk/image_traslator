import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# 기본 설정
DEFAULT_SOURCE_LANG = 'en'
DEFAULT_TARGET_LANG = 'ko'
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# OCR 설정
OCR_SETTINGS = {
    'use_angle_cls': True,
    'det_db_thresh': 0.3,
    'det_db_box_thresh': 0.6,
    'det_db_unclip_ratio': 1.5,
    'rec_batch_num': 6
}

# 번역 엔진 설정
TRANSLATION_ENGINES = {
    'google': 'googletrans',
    'deep_translator': 'deep_translator'
}

# 폰트 설정
FONT_PATHS = {
    'korean': str(PROJECT_ROOT / 'fonts' / 'NanumGothic.ttf'),
    'english': str(PROJECT_ROOT / 'fonts' / 'arial.ttf'),
    'default': None
}

# 임시 파일 디렉토리
TEMP_DIR = PROJECT_ROOT / 'temp'
TEMP_DIR.mkdir(exist_ok=True)

# 로깅 설정
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'