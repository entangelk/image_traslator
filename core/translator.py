from typing import List, Optional
from googletrans import Translator as GoogleTranslator
from deep_translator import GoogleTranslator as DeepGoogleTranslator
from models.text_block import TextBlock


class TextTranslator:
    def __init__(self, source_lang='auto', target_lang='ko', engine='google'):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.engine = engine
        
        if engine == 'google':
            self.translator = GoogleTranslator()
        elif engine == 'deep_translator':
            # auto 감지는 googletrans로 처리
            if source_lang == 'auto':
                self.translator = GoogleTranslator()
            else:
                self.translator = DeepGoogleTranslator(source=source_lang, target=target_lang)
        else:
            raise ValueError(f"Unsupported translation engine: {engine}")
    
    def translate_text(self, text: str) -> Optional[str]:
        try:
            if self.engine == 'google' or self.source_lang == 'auto':
                # Google Translate는 자동 언어 감지 지원
                src_lang = self.source_lang if self.source_lang != 'auto' else None
                result = self.translator.translate(text, src=src_lang, dest=self.target_lang)
                return result.text
            elif self.engine == 'deep_translator':
                return self.translator.translate(text)
        except Exception as e:
            print(f"Translation error for '{text}': {e}")
            return None
    
    def translate_blocks(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        for block in text_blocks:
            if block.original_text.strip():
                translated = self.translate_text(block.original_text)
                if translated:
                    block.translated_text = translated
        
        return text_blocks
    
    def batch_translate(self, texts: List[str]) -> List[Optional[str]]:
        results = []
        for text in texts:
            results.append(self.translate_text(text))
        return results