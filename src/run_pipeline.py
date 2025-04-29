#!/usr/bin/env python3
"""
run_pipeline.py: Audio feature extraction pipeline with parallelism and lightweight ASR.

- Voice Activity Detection (VAD)
- Speaker Diarization
- Automatic Speech Recognition (ASR) via Whisper
- Overlapped Speech Detection
- Words Per Minute (WPM) calculation
- Text-based Emotion Analysis

Processes all .wav files under provided directories in parallel using ProcessPoolExecutor.
Includes worker initializer for heavy model/pipeline loading and per-worker logging.
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

# add project src directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from load_preprocess import load_audio
import VAD
import overlapped_speech
import speaker_diarization
import ASR
import WPM
from text_based_emotion_recongition import analyze_text_emotion

# Global placeholder for ASR model size
WHISPER_SIZE = None


def init_worker(whisper_size: str):
    """Initializer for each worker: load heavy models/pipelines once."""
    global WHISPER_SIZE
    WHISPER_SIZE = whisper_size
    # Import here to avoid module-level overhead
    import whisper
    from pyannote.audio import Pipeline

    # Load Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ASR._model = whisper.load_model(whisper_size, device=device)

    # Load pyannote pipelines
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
    # Log start
    print(f"[Worker {os.getpid()}] processing {wav_path}")

    # 1) Load audio
    audio, sr = load_audio(wav_path)
    duration = len(audio) / sr

    # 2) VAD
    speech_segments = VAD.detect_speech(wav_path)
    speech_duration = sum(e - s for s, e in speech_segments)
    silence_ratio = (duration - speech_duration) / duration

    # 3) Diarization
    annotation = speaker_diarization._diar_pipeline(wav_path)
    durations = speaker_diarization.diarize(wav_path)

    # 4) ASR
    asr_segs = ASR.transcribe(wav_path)
    transcript = " ".join(seg.text for seg in asr_segs)
    segments_str = "|".join(f"{seg.start:.2f}-{seg.end:.2f}:{seg.text}" for seg in asr_segs)

    # 5) Overlapped Speech
    overlaps = overlapped_speech.detect_overlap(wav_path)
    overlap_duration = sum(e - s for s, e in overlaps)

    # 6) WPM
    wpm = WPM.compute_wpm(asr_segs, annotation)

    # 7) Emotion
    emo = analyze_text_emotion(transcript, top_k=3)

    # Build result row
    row = {
        "file": wav_path,
        "duration_s": duration,
        "speech_count": len(speech_segments),
        "speech_s": speech_duration,
        "silence_ratio": silence_ratio,
        "num_speakers": len(durations),
        "speaker_durations": "|".join(f"{spk}:{dur:.2f}" for spk, dur in durations.items()),
        "transcript": transcript,
        "asr_segments": segments_str,
        "overlap_count": len(overlaps),
        "overlap_s": overlap_duration,
        "overlap_ratio": overlap_duration / duration,
        "wpm": "|".join(f"{spk}:{rate:.1f}" for spk, rate in wpm.items()),
        "sent_label": emo.get("label", ""),
        "sent_score": emo.get("score", 0.0),
        "top_nouns": "|".join(f"{n}:{c}" for n, c in emo.get("top_nouns", []))
    }
    for lbl, sc in emo.get("distribution", {}).items():
        row[f"emo_{lbl}_sc"] = sc
    return row


def collect_paths(dirs: list) -> list:
    paths = []
    for d in dirs:
        paths.extend(glob.glob(os.path.join(d, "S*", "*.wav")))
    return paths


def main(args):
    # Parallel worker initializer
    executor_kwargs = {
        'max_workers': args.workers,
        'initializer': init_worker,
        'initargs': (args.whisper_size,)
    }

    paths = collect_paths(args.dirs)
    print(f"Found {len(paths)} files; workers={args.workers}, Whisper='{args.whisper_size}'")

    # Execute in parallel with initializer
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        rows = list(tqdm(
            executor.map(process_file, paths),
            total=len(paths), desc="Processing files"
        ))

    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel audio feature extraction with worker init and logging"
    )
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="Root directories containing S*/.wav files")
    parser.add_argument("--output", default="audio_features.csv",
                        help="Output CSV filename")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel workers")
    parser.add_argument("--whisper-size", default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size for ASR")
    args = parser.parse_args()
    main(args)
