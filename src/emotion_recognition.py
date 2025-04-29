# --- 7. Emotion Recognition (Detailed) ---
'''
superb/wav2vec2-base-superb-er 모델을 사용하여 감정 인식
(top_k=3 후보 & 화자별 확률 분포까지)
'''

import librosa
from pyannote.audio import Pipeline
from transformers import pipeline
from collections import defaultdict

# ———————————————————————————————
# 설정
HF_TOKEN = True  # Hugging Face 토큰을 환경변수나 로그인된 상태에서 사용 중이라면 True
wav_path = "../data/consult_voice/Training/raw/KtelSpeech_train_D62_wav_0/D62/J93/S00000001/0002.wav"

# ———————————————————————————————
# 1) 화자 분리
diar = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)
diar_result     = diar(wav_path)
speech_timeline = diar_result.get_timeline()

# ———————————————————————————————
# 2) 감정 분류기 로드
clf = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device=-1
)

# ———————————————————————————————
# 3) 오디오 로드 & 모노 + 16 kHz 리샘플링
signal, sr = librosa.load(wav_path, sr=16000, mono=True)

# ———————————————————————————————
# 4) 화자별 감정 집계 구조
speaker_emotion_counts = defaultdict(lambda: defaultdict(int))
speaker_emotion_probs  = defaultdict(list)

# ———————————————————————————————
# 5) 세그먼트별 인식 & 화자 매핑
for idx, seg in enumerate(speech_timeline, 1):
    # 5-1) 화자 매핑 (겹친 시간만큼 계산)
    overlaps = {}
    for turn, _, spk in diar_result.itertracks(yield_label=True):
        overlap = max(0, min(turn.end, seg.end) - max(turn.start, seg.start))
        if overlap > 0:
            overlaps[spk] = overlap
    speaker = max(overlaps, key=overlaps.get) if overlaps else "UNKNOWN"

    # 5-2) 오디오 슬라이스
    start_sample = int(seg.start * sr)
    end_sample   = int(seg.end   * sr)
    chunk        = signal[start_sample:end_sample]

    # 5-3) top_k=3 후보 감정 분류
    results = clf(chunk, sampling_rate=sr, top_k=3)

    # 5-4) top1 빈도 카운트
    top1 = results[0]["label"]
    speaker_emotion_counts[speaker][top1] += 1

    # 5-5) 후보 점수 저장
    speaker_emotion_probs[speaker].append(results)

# ———————————————————————————————
# 6) 화자별 누적 확률 계산
speaker_prob_sum = defaultdict(lambda: defaultdict(float))
for spk, seg_results in speaker_emotion_probs.items():
    for results in seg_results:
        for r in results:
            speaker_prob_sum[spk][r["label"]] += r["score"]

# ———————————————————————————————
# 7) 결과 출력

# 7-1) 단순 빈도 기반 분포
print("\n🎭 화자별 top1 감정 분포 (빈도)\n")
for spk, emo_cnt in speaker_emotion_counts.items():
    total = sum(emo_cnt.values())
    print(f"🔈 {spk} (총 세그먼트: {total})")
    for label, cnt in emo_cnt.items():
        pct = cnt / total * 100
        print(f"  • {label:<8}: {cnt}개 ({pct:.1f}%)")
    print()

# 7-2) 세그먼트별 후보 & 점수
print("🔍 세그먼트별 감정 후보 (top3)\n")
for spk, seg_results in speaker_emotion_probs.items():
    print(f"🔈 {spk}:")
    for i, results in enumerate(seg_results, 1):
        candidates = ", ".join(f"{r['label']}({r['score']:.2f})" for r in results)
        print(f"  • 세그먼트 {i}: {candidates}")
    print()

# 7-3) 누적 확률 기반 분포
print("📊 화자별 누적 확률 분포\n")
for spk, prob_dict in speaker_prob_sum.items():
    total_score = sum(prob_dict.values())
    print(f"🔈 {spk}:")
    for label, score in prob_dict.items():
        pct = score / total_score * 100
        print(f"  • {label:<8}: 총점={score:.2f}  ({pct:.1f}%)")
    print()

'''
🎭 화자별 top1 감정 분포 (빈도)

🔈 SPEAKER_00 (총 세그먼트: 1)
  • neu     : 1개 (100.0%)

🔍 세그먼트별 감정 후보 (top3)

🔈 SPEAKER_00:
  • 세그먼트 1: neu(0.95), hap(0.05), sad(0.00)

📊 화자별 누적 확률 분포

🔈 SPEAKER_00:
  • neu     : 총점=0.95  (94.7%)
  • hap     : 총점=0.05  (5.1%)
  • sad     : 총점=0.00  (0.2%)
'''

'''
빈도(top1): 각 세그먼트에서 가장 자신 있는 감정만 단순 세기
후보(top3): 한 구간 안에서 모델이 본 세 가지 가능성과 확률
누적 확률: 모든 후보 확률을 합산해, 여러 구간에 걸친 감정 분포를 부드럽게 표현
'''