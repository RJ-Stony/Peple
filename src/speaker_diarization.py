# --- 3. Speaker Diarization ---
'''
https://huggingface.co/pyannote/speaker-diarization에서 Speaker Diarization 모델을 다운로드하여 사용

OSError: [WinError 1314] 클라이언트가 필요한 권한을 가지고 있지 않습니다:
Windows에서 Hugging Face 허브가 기본으로 심볼릭 링크(symlink)를 쓰도록 되어 있는데, 
현재 권한(Developer Mode 미활성화 또는 비관리자 권한)으로는 symlink 생성이 막혀 있기 때문에 발생
-> VSCODE 관리자 권한으로 실행 후 다시 코드 실행
'''
from typing import Dict
from pyannote.audio import Pipeline
from collections import defaultdict

# Speaker Diarization pipeline 로드 (HuggingFace 토큰 필요)
_diar_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=True
)

def diarize(wav_path: str) -> Dict[str, float]:
    """
    주어진 WAV 파일에 대해 화자 분리(Diarization)를 수행하고,
    각 화자별 발화 총합(초) 딕셔너리를 반환합니다.

    Args:
        wav_path (str): 분석할 오디오 파일 경로(.wav)

    Returns:
        Dict[str, float]: {화자_레이블: 발화시간_초}
    """
    # pipeline 실행
    annotation = _diar_pipeline(wav_path)
    # 각 화자별 발화 시간 합산
    durations = defaultdict(float)
    # itertracks: (segment, track_id, speaker_label)
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        durations[speaker] += segment.end - segment.start
    return dict(durations)
