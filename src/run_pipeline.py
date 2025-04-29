"""
run_pipeline.py: Audio feature extraction pipeline

- Voice Activity Detection (VAD)
- Speaker Diarization
- Automatic Speech Recognition (ASR)
- Overlapped Speech Detection
- Words Per Minute (WPM) calculation
- Text-based Emotion Analysis

Results for each .wav file are saved as a single row in an output CSV.
"""
import os
import sys
import glob
import csv
import argparse
from tqdm import tqdm
import pandas as pd

# add project src directory if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from load_preprocess import load_audio
import VAD
import overlapped_speech
import speaker_diarization
import ASR
import WPM
from text_based_emotion_recongition import analyze_text_emotion


def process_file(wav_path: str) -> dict:
    # 1) Load audio
    audio, sr = load_audio(wav_path)
    duration_seconds = len(audio) / sr

    # 2) Voice Activity Detection
    speech_segments = VAD.detect_speech(wav_path)
    speech_segment_count = len(speech_segments)
    speech_duration_seconds = sum(end - start for start, end in speech_segments)
    silence_ratio = (duration_seconds - speech_duration_seconds) / duration_seconds

    # 3) Speaker Diarization
    # annotation object for WPM
    annotation = speaker_diarization._diar_pipeline(wav_path)
    # durations dict for reporting
    durations_dict = speaker_diarization.diarize(wav_path)
    speaker_count = len(durations_dict)

    # 4) Automatic Speech Recognition
    asr_segments = ASR.transcribe(wav_path)
    transcript = " ".join(seg.text for seg in asr_segments)
    segment_transcripts = "|".join(f"{seg.start:.2f}-{seg.end:.2f}:{seg.text}" for seg in asr_segments)

    # 5) Overlapped Speech Detection
    overlap_segments = overlapped_speech.detect_overlap(wav_path)
    overlap_segment_count = len(overlap_segments)
    overlap_duration_seconds = sum(end - start for start, end in overlap_segments)
    overlap_ratio = overlap_duration_seconds / duration_seconds

    # 6) Words Per Minute
    wpm_dict = WPM.compute_wpm(asr_segments, annotation)

    # 7) (Optional steps can be inserted here)

    # 8) Text-based Emotion Analysis
    emo = analyze_text_emotion(transcript, top_k=5)
    sentiment_label = emo["label"]
    sentiment_score = emo["score"]
    top_nouns = emo.get("top_nouns", [])
    distribution = emo.get("distribution", {})

    # Build result row
    row = {
        "file_path": wav_path,
        "duration_seconds": duration_seconds,
        "speech_segment_count": speech_segment_count,
        "speech_duration_seconds": speech_duration_seconds,
        "silence_ratio": silence_ratio,
        "speaker_count": speaker_count,
        "speaker_durations": "|".join(f"{spk}:{dur:.2f}" for spk, dur in durations_dict.items()),
        "transcript": transcript,
        "segment_transcripts": segment_transcripts,
        "overlap_segment_count": overlap_segment_count,
        "overlap_duration_seconds": overlap_duration_seconds,
        "overlap_ratio": overlap_ratio,
        "wpm_by_speaker": "|".join(f"{spk}:{wpm:.2f}" for spk, wpm in wpm_dict.items()),
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "top20_nouns": "|".join(f"{noun}:{count}" for noun, count in top_nouns),
    }
    # add emotion distribution columns
    for label, score in distribution.items():
        col = f"emotion_{label}_score"
        row[col] = score

    return row


def main(input_dirs: list, output_csv: str):
    # collect all WAV file paths under each root dir
    wav_paths = []
    for d in input_dirs:
        pattern = os.path.join(d, "S*", "*.wav")
        wav_paths.extend(glob.glob(pattern))
    print(f"Found {len(wav_paths)} WAV files to process.")

    results = []
    for wav in tqdm(wav_paths, desc="Processing audio files"):  # progress bar
        try:
            results.append(process_file(wav))
        except Exception as e:
            print(f"[ERROR] {wav}: {e}")

    # save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract audio features and save to CSV"
    )
    parser.add_argument(
        "--dirs", nargs='+', required=True,
        help="Root directories containing 'S*' subfolders with WAV files"
    )
    parser.add_argument(
        "--output", default="audio_features.csv",
        help="Output CSV filename (default: audio_features.csv)"
    )
    args = parser.parse_args()
    main(args.dirs, args.output)