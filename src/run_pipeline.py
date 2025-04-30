#!/usr/bin/env python3
"""
run_pipeline.py: Audio feature extraction pipeline with parallelism, reliable CSV output,
and inclusion of ground-truth label transcripts.

- Voice Activity Detection (VAD)
- Speaker Diarization
- Automatic Speech Recognition (ASR) via Whisper
- Overlapped Speech Detection
- Words Per Minute (WPM) calculation
- Text-based Emotion Analysis
- Loading and cleaning original transcript from label files

Results for each .wav file are saved as rows in an output CSV. This version:
  • Adds 'label_transcript' from corresponding .txt (cleans slashes and parentheses)
  • Fills missing emotion scores with 0, empty noun lists with '-'
  • Writes CSV with utf-8-sig for proper Korean handling
  • Uses a worker initializer to load heavy models once per process
  • Logs each worker's initialization and per-file start
"""
import os
import sys
import glob
import csv
import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import torch

# add project src directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from load_preprocess import load_audio
import VAD
import overlapped_speech
import speaker_diarization
import ASR
import WPM
from text_based_emotion_recongition import analyze_text_emotion

# Fixed set of emotion labels for consistency
EMOTION_LABELS = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']

# Global whisper model size placeholder
WHISPER_SIZE = None


def init_worker(whisper_size: str):
    """Worker initializer: load Whisper and pyannote pipelines once."""
    global WHISPER_SIZE
    WHISPER_SIZE = whisper_size
    import whisper
    from pyannote.audio import Pipeline

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ASR._model = whisper.load_model(whisper_size, device=device)

    VAD._vad_pipeline = Pipeline.from_pretrained(
        'pyannote/voice-activity-detection', use_auth_token=True
    )
    overlapped_speech._ovlp_pipeline = Pipeline.from_pretrained(
        'pyannote/overlapped-speech-detection', use_auth_token=True
    )
    speaker_diarization._diar_pipeline = Pipeline.from_pretrained(
        'pyannote/speaker-diarization', use_auth_token=True
    )

    print(f"[Worker {os.getpid()}] initialized (Whisper={whisper_size}, device={device})")


def process_file(wav_path: str) -> dict:
    """Process a single wav: extract features and load cleaned label transcript."""
    print(f"[Worker {os.getpid()}] processing {wav_path}")

    # 1) Load audio
    audio, sr = load_audio(wav_path)
    duration = len(audio) / sr

    # 2) VAD
    speech_segments = VAD.detect_speech(wav_path)
    speech_duration = sum(e - s for s, e in speech_segments)
    silence_ratio = (duration - speech_duration) / duration if duration > 0 else 0.0

    # 3) Diarization
    annotation = speaker_diarization._diar_pipeline(wav_path)
    durations_dict = speaker_diarization.diarize(wav_path)

    # 4) ASR
    asr_segments = ASR.transcribe(wav_path)
    transcript = ' '.join(seg.text for seg in asr_segments)
    segments_str = '|'.join(f"{seg.start:.2f}-{seg.end:.2f}:{seg.text}" for seg in asr_segments)

    # 4.5) Load and clean original label transcript if exists
    # Derive label path by replacing 'raw'->'label', folder name, and extension
    parts = os.path.normpath(wav_path).split(os.sep)
    label_transcript = ''
    if 'raw' in parts:
        idx = parts.index('raw')
        parts[idx] = 'label'
        parts[idx+1] = parts[idx+1].replace('_wav_', '_label_')
        # change filename extension
        parts[-1] = os.path.splitext(parts[-1])[0] + '.txt'
        label_path = os.sep.join(parts)
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                raw = f.read()
            # remove parentheses content, slashes, newlines
            cleaned = re.sub(r"\([^)]*\)", '', raw)
            cleaned = cleaned.replace('/', ' ').replace('\n', ' ')
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            label_transcript = cleaned

    # 5) Overlapped speech
    overlap_segments = overlapped_speech.detect_overlap(wav_path)
    overlap_duration = sum(e - s for s, e in overlap_segments)

    # 6) WPM
    wpm_dict = WPM.compute_wpm(asr_segments, annotation)

    # 7) Emotion analysis
    emo = analyze_text_emotion(label_transcript, top_k=5)
    top_nouns = emo.get('top_nouns', [])
    top_nouns_str = '|'.join(f"{n}:{c}" for n, c in top_nouns) if top_nouns else '-'

    # Build row dict
    row = {
        'file': wav_path,
        'duration_s': duration,
        'speech_count': len(speech_segments),
        'speech_s': speech_duration,
        'silence_ratio': silence_ratio,
        'num_speakers': len(durations_dict),
        'speaker_durations': '|'.join(f"{spk}:{dur:.2f}" for spk, dur in durations_dict.items()),
        'transcript': transcript,
        'asr_segments': segments_str,
        'label_transcript': label_transcript,
        'overlap_count': len(overlap_segments),
        'overlap_s': overlap_duration,
        'overlap_ratio': overlap_duration / duration if duration > 0 else 0.0,
        'wpm': '|'.join(f"{spk}:{rate:.1f}" for spk, rate in wpm_dict.items()),
        'sent_label': emo.get('label', ''),
        'sent_score': emo.get('score', 0.0),
        'top_nouns': top_nouns_str,
    }
    # Ensure all emotion score columns
    dist = emo.get('distribution', {})
    for lbl in EMOTION_LABELS:
        key = f"emo_{lbl.replace(' ', '_')}_score"
        row[key] = dist.get(lbl, 0.0)

    return row


def collect_paths(dirs: list) -> list:
    """Collect wav paths: direct files, any wavs in dir, or S* subfolders"""
    paths = []
    for d in dirs:
        if os.path.isfile(d) and d.lower().endswith('.wav'):
            paths.append(d)
        else:
            wavs = glob.glob(os.path.join(d, '*.wav'))
            if wavs:
                paths.extend(wavs)
            else:
                paths.extend(glob.glob(os.path.join(d, 'S*', '*.wav')))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Audio pipeline with label transcripts")
    parser.add_argument('--dirs', nargs='+', required=True, help='Files or dirs')
    parser.add_argument('--output', default='audio_features.csv', help='CSV out')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Parallel workers')
    parser.add_argument('--whisper-size', choices=['tiny','base','small','medium','large'], default='small', help='Whisper size')
    args = parser.parse_args()

    paths = collect_paths(args.dirs)
    print(f"Found {len(paths)} files; workers={args.workers}, Whisper={args.whisper_size}")

    executor_kwargs = {'max_workers': args.workers, 'initializer': init_worker, 'initargs': (args.whisper_size,)}
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        rows = list(tqdm(executor.map(process_file, paths), total=len(paths), desc='Processing'))

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
    print(f"Saved results to {args.output}")

if __name__ == '__main__':
    main()
