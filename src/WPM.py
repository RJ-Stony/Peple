# --- 6. Speaker Diarization + ASR (Whisper) ---

from typing import List, Dict, Any
from collections import defaultdict
from pyannote.core import Annotation

# WPM 계산을 위한 함수
# asr_segments: Whisper ASR 결과 세그먼트 리스트(Segment with .start, .end, .text)
# diarization: pyannote.audio Pipeline 실행 결과 Annotation 객체

def compute_wpm(asr_segments: List[Any], diarization: Annotation) -> Dict[str, float]:
    """
    주어진 ASR 세그먼트와 화자 분리 결과를 기반으로, 화자별 분당 단어 수(WPM)를 계산합니다.

    Args:
        asr_segments (List[Any]): start, end 시간과 텍스트를 가진 ASR 세그먼트 객체 리스트.
        diarization (Annotation): 화자 분리 파이프라인의 결과 Annotation 객체.

    Returns:
        Dict[str, float]: {speaker_label: WPM}
    """
    # 화자별 단어 수 및 발화 시간 초기화
    speaker_word_counts: Dict[str, int] = defaultdict(int)
    speaker_durations: Dict[str, float] = defaultdict(float)

    # 1) 화자별 발화 시간 합산
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        speaker_durations[speaker] += segment.end - segment.start

    # 2) ASR 세그먼트별 단어를 가장 많이 겹치는 화자에게 할당
    for seg in asr_segments:
        # ASR 세그먼트 정보
        seg_start, seg_end, text = seg.start, seg.end, seg.text
        # 각 화자와의 겹침 시간 계산
        overlaps: Dict[str, float] = {}
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            overlap = min(seg_end, segment.end) - max(seg_start, segment.start)
            if overlap > 0:
                overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
        if overlaps:
            # 최대 겹침 화자에게 단어 수 할당
            dominant = max(overlaps, key=overlaps.get)
            speaker_word_counts[dominant] += len(text.split())

    # 3) 화자별 WPM 계산
    wpm_rates: Dict[str, float] = {}
    for speaker, count in speaker_word_counts.items():
        duration_sec = speaker_durations.get(speaker, 0.0)
        if duration_sec > 0:
            # WPM = (단어 수 / 발화 시간(분))
            wpm_rates[speaker] = (count / (duration_sec / 60))
        else:
            wpm_rates[speaker] = 0.0

    return wpm_rates
