#!/usr/bin/env python3
"""
run_pipeline.py: Audio feature extraction pipeline with parallelism and reliable CSV output.

- Voice Activity Detection (VAD)
- Speaker Diarization
- Automatic Speech Recognition (ASR) via Whisper
- Overlapped Speech Detection
- Words Per Minute (WPM) calculation
- Text-based Emotion Analysis

Results for each .wav file are saved as rows in an output CSV. This version:
  • Ensures all emotion score columns are present (fills missing with 0)
  • Uses a placeholder for empty noun lists ("-")
  • Writes CSV with UTF-8 BOM to preserve Korean in Excel
  • Uses a worker initializer to load heavy models once per process
  • Logs each worker's initialization and per-file start
"""
import os
import sys
import glob
import csv
import argparse
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

# Fixed set of emotion labels to cover all possible classes
EMOTION_LABELS = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']

# Global placeholder for whisper model size
WHISPER_SIZE = None


def init_worker(whisper_size: str):
    """Initializer for each worker: loads Whisper and pyannote pipelines once."""
    global WHISPER_SIZE
    WHISPER_SIZE = whisper_size
    import whisper
    from pyannote.audio import Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ASR._model = whisper.load_model(whisper_size, device=device)

    VAD._vad_pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection", use_auth_token=True
    )
    overlapped_speech._ovlp_pipeline = Pipeline.from_pretrained(
        "pyannote/overlapped-speech-detection", use_auth_token=True
    )
    speaker_diarization._diar_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=True
    )

    print(f"[Worker {os.getpid()}] initialized with Whisper='{whisper_size}' on {device}")


def process_file(wav_path: str) -> dict:
    """Process a single WAV file end-to-end and return dict of features."""
    print(f"[Worker {os.getpid()}] processing {wav_path}")

    # 1) Load audio
    audio, sr = load_audio(wav_path)
    duration = len(audio) / sr

    # 2) Voice Activity Detection
    speech_segments = VAD.detect_speech(wav_path)
    speech_duration = sum(e - s for s, e in speech_segments)
    silence_ratio = (duration - speech_duration) / duration

    # 3) Speaker Diarization
    annotation = speaker_diarization._diar_pipeline(wav_path)
    durations_dict = speaker_diarization.diarize(wav_path)

    # 4) Automatic Speech Recognition
    asr_segments = ASR.transcribe(wav_path)
    transcript = " ".join(seg.text for seg in asr_segments)
    segments_str = "|".join(f"{seg.start:.2f}-{seg.end:.2f}:{seg.text}" for seg in asr_segments)

    # 5) Overlapped Speech Detection
    overlap_segments = overlapped_speech.detect_overlap(wav_path)
    overlap_duration = sum(e - s for s, e in overlap_segments)

    # 6) Words Per Minute
    wpm_dict = WPM.compute_wpm(asr_segments, annotation)

    # 7) Text-based Emotion Analysis
    emo = analyze_text_emotion(transcript, top_k=5)
    top_nouns_list = emo.get("top_nouns", [])
    top_nouns_str = (
        "|".join(f"{noun}:{cnt}" for noun, cnt in top_nouns_list)
        if top_nouns_list else "-"
    )

    # Build result row
    row = {
        "file": wav_path,
        "duration_s": duration,
        "speech_count": len(speech_segments),
        "speech_s": speech_duration,
        "silence_ratio": silence_ratio,
        "num_speakers": len(durations_dict),
        "speaker_durations": "|".join(f"{spk}:{dur:.2f}" for spk, dur in durations_dict.items()),
        "transcript": transcript,
        "asr_segments": segments_str,
        "overlap_count": len(overlap_segments),
        "overlap_s": overlap_duration,
        "overlap_ratio": overlap_duration / duration,
        "wpm": "|".join(f"{spk}:{rate:.1f}" for spk, rate in wpm_dict.items()),
        "sent_label": emo.get("label", ""),
        "sent_score": emo.get("score", 0.0),
        "top_nouns": top_nouns_str,
    }

    # Ensure all emotion columns are present
    distribution = emo.get("distribution", {})
    for label in EMOTION_LABELS:
        key = f"emo_{label.replace(' ', '_')}_score"
        row[key] = distribution.get(label, 0.0)

    return row


def collect_paths(dirs: list) -> list:
    """
    Collect WAV paths for testing:
      - direct .wav file
      - any .wav files in given dir
      - or fallback to S* subfolders
    """
    paths = []
    for d in dirs:
        if os.path.isfile(d) and d.lower().endswith('.wav'):
            paths.append(d)
            continue
        wavs = glob.glob(os.path.join(d, '*.wav'))
        if wavs:
            paths.extend(wavs)
            continue
        paths.extend(glob.glob(os.path.join(d, 'S*', '*.wav')))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Parallel audio pipeline with robust CSV output"
    )
    parser.add_argument(
        "--dirs", nargs='+', required=True,
        help="List of files or directories to process"
    )
    parser.add_argument(
        "--output", default="audio_features.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--whisper-size", choices=["tiny","base","small","medium","large"],
        default="small",
        help="Size of Whisper ASR model"
    )
    args = parser.parse_args()

    paths = collect_paths(args.dirs)
    print(f"Found {len(paths)} files; workers={args.workers}, Whisper='{args.whisper_size}'")

    executor_kwargs = {
        'max_workers': args.workers,
        'initializer': init_worker,
        'initargs': (args.whisper_size,)
    }

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        rows = list(tqdm(
            executor.map(process_file, paths),
            total=len(paths), desc="Processing files"
        ))

    df = pd.DataFrame(rows)
    
    write_header = not os.path.exists(args.output)
    df.to_csv(
        args.output,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding='utf-8-sig',
        mode='a',
        header=write_header
    )
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
