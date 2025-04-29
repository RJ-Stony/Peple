# --- 4. ASR (Automatic Speech Recognition) ---

'''
FileNotFoundError: [WinError 2] 지정된 파일을 찾을 수 없습니다:
Anaconda Prompt에서 아래 명령어로 ffmpeg 설치 후 다시 실행
conda install -c conda-forge ffmpeg

혹은

Windows Powershell에서 아래 명령어로 ffmpeg 설치 후 다시 실행
choco install ffmpeg -y

whisper 라이브러리 설치
pip install -U openai-whisper
'''
# Whisper 모델 로드 (tiny/base/small/medium/large 중 선택)
# base 모델은 GB, small 모델은 GB, medium 모델은 1.42GB, large 모델은 32GB
# base 모델은 인식을 잘 못해서 small 모델 추천
import whisper

from dataclasses import dataclass
from typing import List

# Whisper 모델 로드 (모델 크기는 필요에 따라 조정하세요: tiny/base/small/medium/large)
_model = whisper.load_model("small")

@dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe(wav_path: str) -> List[Segment]:
    """
    주어진 WAV 파일을 Whisper 모델로 ASR 수행 후, 세그먼트별 텍스트와 시간 정보를 반환합니다.

    Args:
        wav_path (str): 분석할 오디오 파일 경로(.wav)

    Returns:
        List[Segment]: start(초), end(초), text(문장) 정보를 담은 리스트
    """
    # Whisper로 음성 인식 실행 (한국어 지정, 세그먼트별 시간 포함)
    result = _model.transcribe(
        wav_path,
        language="ko",
        word_timestamps=False  # 세그먼트별로 시간 제공
    )

    segments: List[Segment] = []
    for seg in result.get("segments", []):
        segments.append(
            Segment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip()
            )
        )
    return segments
