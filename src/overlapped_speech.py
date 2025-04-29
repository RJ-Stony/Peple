# --- 5. Overlapped Speech Detection ---
'''
https://huggingface.co/pyannote/overlapped-speech-detection 에서 Overlapped Speech Detection 모델을 다운로드하여 사용
'''
from typing import List, Tuple
from pyannote.audio import Pipeline

# Overlapped Speech Detection pipeline 로드 (HuggingFace 토큰 필요)
_ovlp_pipeline = Pipeline.from_pretrained(
    "pyannote/overlapped-speech-detection",
    use_auth_token=True
)

def detect_overlap(wav_path: str) -> List[Tuple[float, float]]:
    """
    주어진 WAV 파일에서 겹쳐 말하기(overlap) 구간을 탐지하여
    (start, end) 리스트로 반환합니다.

    Args:
        wav_path (str): 분석할 오디오 파일 경로(.wav)

    Returns:
        List[Tuple[float, float]]: 겹쳐 말하기 구간의 시작·끝 시간(초) 리스트
    """
    # 파이프라인 실행
    result = _ovlp_pipeline(wav_path)
    timeline = result.get_timeline()
    segments: List[Tuple[float, float]] = []
    for segment in timeline:
        segments.append((segment.start, segment.end))
    return segments
