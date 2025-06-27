from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TextStyle:
    font_family: Optional[str] = None
    font_size: Optional[int] = None
    color: Optional[Tuple[int, int, int]] = None
    background_color: Optional[Tuple[int, int, int]] = None
    bold: bool = False
    italic: bool = False


@dataclass
class TextBlock:
    x: int
    y: int
    width: int
    height: int
    original_text: str
    translated_text: Optional[str] = None
    confidence: float = 0.0
    style: Optional[TextStyle] = None
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)