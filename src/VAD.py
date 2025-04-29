# --- 2. Voice Activity Detection (VAD) ---
'''
pyannote/segmentation https://huggingface.co/pyannote/segmentation에서 Segmentation 모델을 다운로드하여 사용
pyannote/voice-activity-detection https://huggingface.co/pyannote/voice-activity-detection에서 VAD 모델을 다운로드하여 사용
'''
'''
pip install huggingface-hub
huggingface-cli login 후 HF_TOKEN을 입력하여 로그인
'''

from typing import List, Tuple
from pyannote.audio import Pipeline

# Voice Activity Detection pipeline 로드 (HuggingFace 토큰 필요)
_vad_pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=True
)

def detect_speech(wav_path: str) -> List[Tuple[float, float]]:
    """
    주어진 WAV 파일에 대해 VAD를 수행하고, 발화(음성) 구간 리스트를 반환합니다.

    Args:
        wav_path (str): 분석할 오디오 파일 경로(.wav)

    Returns:
        List[Tuple[float, float]]: (start, end) 형식의 발화 구간 시각(초) 리스트
    """
    # 파이프라인 실행
    result = _vad_pipeline(wav_path)
    # 타임라인 객체 얻기
    timeline = result.get_timeline()
    # Timeline을 순회하며 구간(start, end)을 리스트에 저장
    segments: List[Tuple[float, float]] = []
    for segment in timeline:
        segments.append((segment.start, segment.end))
    return segments
