import cv2
import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from models.text_block import TextBlock, TextStyle


class StyleAnalyzer:
    def __init__(self):
        pass
    
    def analyze_text_style(self, image: np.ndarray, text_block: TextBlock) -> TextStyle:
        x, y, w, h = text_block.x, text_block.y, text_block.width, text_block.height
        roi = image[y:y+h, x:x+w]
        
        # 색상 분석
        text_color = self._extract_text_color(roi)
        bg_color = self._extract_background_color(roi)
        
        # 폰트 크기 추정
        font_size = self._estimate_font_size(text_block)
        
        # 폰트 스타일 분석 (굵기, 기울임)
        is_bold, is_italic = self._analyze_font_style(roi)
        
        return TextStyle(
            font_size=font_size,
            color=text_color,
            background_color=bg_color,
            bold=is_bold,
            italic=is_italic
        )
    
    def _extract_text_color(self, roi: np.ndarray) -> Tuple[int, int, int]:
        # 그레이스케일로 변환
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 텍스트는 보통 더 어두운 색이므로 하위 percentile 사용
        text_intensity = np.percentile(gray, 15)
        
        # 해당 강도에 가까운 픽셀들의 평균 색상 추출
        mask = np.abs(gray - text_intensity) < 30
        if np.any(mask):
            text_pixels = roi[mask]
            return tuple(map(int, np.mean(text_pixels, axis=0)))
        
        return (0, 0, 0)  # 기본 검은색
    
    def _extract_background_color(self, roi: np.ndarray) -> Optional[Tuple[int, int, int]]:
        # 배경은 보통 더 밝은 색이므로 상위 percentile 사용
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bg_intensity = np.percentile(gray, 85)
        
        # 배경과 텍스트의 대비가 충분한지 확인
        text_intensity = np.percentile(gray, 15)
        if bg_intensity - text_intensity < 50:
            return None  # 대비가 충분하지 않으면 배경색 없음
        
        # 해당 강도에 가까운 픽셀들의 평균 색상 추출
        mask = np.abs(gray - bg_intensity) < 30
        if np.any(mask):
            bg_pixels = roi[mask]
            return tuple(map(int, np.mean(bg_pixels, axis=0)))
        
        return None
    
    def _estimate_font_size(self, text_block: TextBlock) -> int:
        # 텍스트 블록의 높이를 기반으로 폰트 크기 추정
        # 일반적으로 폰트 크기는 텍스트 높이의 70-80% 정도
        estimated_size = int(text_block.height * 0.75)
        
        # 최소/최대 크기 제한
        return max(8, min(72, estimated_size))
    
    def _analyze_font_style(self, roi: np.ndarray) -> Tuple[bool, bool]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 굵기 분석: 에지의 두께를 측정
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # 에지가 굵으면 볼드체일 가능성
        edge_ratio = np.sum(dilated > 0) / np.sum(edges > 0) if np.sum(edges > 0) > 0 else 1
        is_bold = edge_ratio > 1.3
        
        # 기울임 분석: 수직선의 기울기 측정
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=min(30, roi.shape[1]//3))
        is_italic = False
        
        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                # 수직에 가까운 선들만 고려
                if 80 < angle < 100:
                    angles.append(angle)
            
            if angles:
                mean_angle = np.mean(angles)
                # 수직에서 많이 벗어나면 기울임체
                is_italic = abs(mean_angle - 90) > 5
        
        return is_bold, is_italic
    
    def cluster_similar_styles(self, text_blocks: List[TextBlock]) -> List[List[TextBlock]]:
        if not text_blocks:
            return []
        
        # 스타일 특성을 벡터로 변환
        features = []
        for block in text_blocks:
            if block.style:
                feature = [
                    block.style.font_size or 16,
                    block.style.color[0] if block.style.color else 0,
                    block.style.color[1] if block.style.color else 0,
                    block.style.color[2] if block.style.color else 0,
                    int(block.style.bold),
                    int(block.style.italic)
                ]
                features.append(feature)
            else:
                features.append([16, 0, 0, 0, 0, 0])
        
        # K-means 클러스터링
        n_clusters = min(5, len(text_blocks))  # 최대 5개 클러스터
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        
        # 클러스터별로 그룹화
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(text_blocks[i])
        
        return [cluster for cluster in clusters if cluster]