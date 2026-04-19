"""
YouTube → Diarized Transcript (with checkpointing)
----------------------------------------------------
Downloads audio, transcribes, diarizes, and saves output in A:/B: format.
Each step saves its output so if something crashes you can resume from
where you left off without rerunning completed steps.

Checkpoints saved:
    debate_audio.mp3           ← Step 1 output (audio)
    checkpoint_transcript.json ← Step 2 output (raw whisper segments)
    checkpoint_aligned.json    ← Step 3 output (diarized segments)
    transcript_labeled.txt     ← Final output

To resume after a crash: just run the script again.
To force rerun a step: delete its checkpoint file and rerun.

Prerequisites: See INSTALL.txt
"""

import os
import sys
import json
import subprocess
import re

import torch
# Fix for PyTorch 2.6 weights_only default change
import torch.serialization
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

# Monkey-patch torch.load to use weights_only=False for pyannote compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False      # ← force False, don't use setdefault
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


# ── EDIT THESE ──────────────────────────────────────────────────────────────────
VIDEO_URL   = "https://www.youtube.com/watch?v=3O6hR8Chsrk"
HF_TOKEN    = "REMOVED_TOKEN"

OUTPUT_FILE    = "transcript_labeled.txt"
AUDIO_FILE     = "debate_audio.mp3"
CKPT_WHISPER   = "checkpoint_transcript.json"   # raw whisper output
CKPT_ALIGNED   = "checkpoint_aligned.json"      # after alignment + diarization

DEVICE         = "cpu"       # "cuda" if you have NVIDIA GPU
WHISPER_MODEL  = "base"      # "small" or "medium" for better accuracy
BROWSER        = "chrome"    # chrome / firefox / edge / brave / safari
# ────────────────────────────────────────────────────────────────────────────────


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"      ✓ Checkpoint saved: {path}")


def load_checkpoint(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_exists(path: str) -> bool:
    return os.path.exists(path)


# ── Step 1: Download audio ────────────────────────────────────────────────────

def step1_download(url: str, output_path: str):
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n[1/4] Audio already exists: {output_path} ({size_mb:.1f} MB) — skipping")
        return

    print(f"\n[1/4] Downloading audio from YouTube...")
    print(f"      URL: {url}")

    command = [
        "yt-dlp",
        "--cookies-from-browser", BROWSER,
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--output", output_path,
        url
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("ERROR: yt-dlp failed:")
        print(result.stderr)
        sys.exit(1)

    if not os.path.exists(output_path):
        print("ERROR: Audio file was not created.")
        sys.exit(1)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      ✓ Downloaded: {output_path} ({size_mb:.1f} MB)")


# ── Step 2: Transcribe ────────────────────────────────────────────────────────

def step2_transcribe(audio_path: str, device: str, model_name: str) -> dict:
    if checkpoint_exists(CKPT_WHISPER):
        print(f"\n[2/4] Whisper checkpoint found — loading {CKPT_WHISPER} (skipping transcription)")
        return load_checkpoint(CKPT_WHISPER)

    print(f"\n[2/4] Transcribing with Whisper ({model_name}) on {device}...")
    print(f"      This may take several minutes on CPU...")

    import whisperx

    compute_type = "float32" if device == "cpu" else "float16"

    model  = whisperx.load_model(model_name, device=device,
                                 compute_type=compute_type, language="en")
    audio  = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=8, language="en")

    print(f"      ✓ Transcribed {len(result['segments'])} segments")
    save_checkpoint(result, CKPT_WHISPER)
    return result


# ── Step 3: Align + Diarize ───────────────────────────────────────────────────

def step3_diarize(audio_path: str, whisper_result: dict, device: str, hf_token: str) -> list:
    if checkpoint_exists(CKPT_ALIGNED):
        print(f"\n[3/4] Diarization checkpoint found — loading {CKPT_ALIGNED} (skipping diarization)")
        data = load_checkpoint(CKPT_ALIGNED)
        return data["segments"]

    print(f"\n[3/4] Aligning and diarizing...")
    print(f"      Downloading models on first run — may take a while...")

    import whisperx

    audio = whisperx.load_audio(audio_path)

    # ── 3a: Word-level alignment ──────────────────────────────────────────────
    print("      Running word alignment...")
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned = whisperx.align(
        whisper_result["segments"], align_model, metadata,
        audio, device=device, return_char_alignments=False
    )

    # ── 3b: Diarization ───────────────────────────────────────────────────────
    print("      Running speaker diarization...")
    diarize_segments = _run_diarization(audio, hf_token, device)

    # ── 3c: Assign speakers ───────────────────────────────────────────────────
    result_with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned)
    segments = result_with_speakers["segments"]

    print(f"      ✓ Done — {len(segments)} segments with speaker labels")
    save_checkpoint({"segments": segments}, CKPT_ALIGNED)
    return segments


def _run_diarization(audio, hf_token: str, device: str):
    import pandas as pd
    from pyannote.audio import Pipeline

    print("      Using pyannote.audio directly...")

    # Newer pyannote (>=3.0) renamed use_auth_token → token
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token                      # ← this is the fix
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token             # ← old version fallback
        )

    pipeline.to(torch.device(device))

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    diarization  = pipeline(
        {"waveform": audio_tensor, "sample_rate": 16000},
        min_speakers=2,
        max_speakers=2
    )

    rows = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        rows.append({
            "start":   turn.start,
            "end":     turn.end,
            "speaker": speaker
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"      ✓ pyannote found {df['speaker'].nunique()} speakers")
    return df


# ── Step 4: Format and save ───────────────────────────────────────────────────

def step4_format(segments: list, output_path: str) -> str:
    print(f"\n[4/4] Formatting and saving transcript...")

    # ── Build speaker map: first appearance order → A, B, C... ───────────────
    seen = []
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        if spk not in seen:
            seen.append(spk)

    labels      = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    speaker_map = {spk: labels[i] for i, spk in enumerate(seen)}

    print("      Speaker mapping:")
    for raw, label in speaker_map.items():
        print(f"        {raw} → {label}")
    print()
    print("      ⚠️  Check the output file to confirm A and B are correct.")
    print("         If swapped, just find-and-replace A↔B in the file.\n")

    # ── Merge consecutive segments from the same speaker ─────────────────────
    merged = []
    for seg in segments:
        raw_spk = seg.get("speaker", "UNKNOWN")
        label   = speaker_map.get(raw_spk, "?")
        text    = seg.get("text", "").strip()

        if not text:
            continue

        if merged and merged[-1]["speaker"] == label:
            merged[-1]["text"] += " " + text
        else:
            merged.append({"speaker": label, "text": text})

    # ── Clean text ────────────────────────────────────────────────────────────
    def clean(text: str) -> str:
        text = re.sub(r'\b(um+|uh+|er+|ah+|hmm+)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s([?.!,])', r'\1', text)
        return text

    # ── Write output in A: / B: format ───────────────────────────────────────
    lines = []
    for turn in merged:
        text = clean(turn["text"])
        if text:
            lines.append(f"{turn['speaker']}: {text}")

    output_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    a_count = sum(1 for l in lines if l.startswith("A:"))
    b_count = sum(1 for l in lines if l.startswith("B:"))

    print(f"      ✓ Saved: {output_path}")
    print(f"        Total turns : {len(lines)}")
    print(f"        Speaker A   : {a_count} turns")
    print(f"        Speaker B   : {b_count} turns")

    return output_text


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_checkpoints():
    """
    Call this manually after everything works to delete intermediate files.
    The final transcript_labeled.txt is NOT deleted.
    """
    for f in [CKPT_WHISPER, CKPT_ALIGNED, AUDIO_FILE]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted: {f}")
    print("Cleanup done. Final output kept:", OUTPUT_FILE)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if "YOUR_VIDEO_ID" in VIDEO_URL:
        print("ERROR: Please set VIDEO_URL before running.")
        sys.exit(1)

    if "YOUR_HUGGINGFACE_TOKEN" in HF_TOKEN:
        print("ERROR: Please set HF_TOKEN before running.")
        print("       Get a free token at https://huggingface.co/settings/tokens")
        sys.exit(1)

    print("=" * 60)
    print("  YouTube → Diarized Transcript")
    print("=" * 60)
    print("  Each step saves a checkpoint.")
    print("  If the script crashes, just run it again to resume.")
    print("  To force-rerun a step, delete its checkpoint file.")
    print("=" * 60)

    # Each step checks for its own checkpoint before running
    step1_download(VIDEO_URL, AUDIO_FILE)
    whisper_result = step2_transcribe(AUDIO_FILE, DEVICE, WHISPER_MODEL)
    segments       = step3_diarize(AUDIO_FILE, whisper_result, DEVICE, HF_TOKEN)
    output         = step4_format(segments, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("  Preview (first 800 chars):")
    print("=" * 60)
    print(output[:800] + ("..." if len(output) > 800 else ""))
    print("=" * 60)
    print(f"\n✅ Done! Output saved to '{OUTPUT_FILE}'")
    print(f"\nIntermediate checkpoint files kept (safe to delete once done):")
    print(f"  {AUDIO_FILE}")
    print(f"  {CKPT_WHISPER}")
    print(f"  {CKPT_ALIGNED}")
    print(f"\nTo delete them: from youtube_to_transcript import cleanup_checkpoints")
    print(f"\nTo feed into preprocessor:")
    print(f"  from preprocessor import run_from_text")
    print(f"  with open('{OUTPUT_FILE}') as f: transcript = f.read()")
    print(f"  result = run_from_text('Messi vs Ronaldo: Who is the GOAT?', transcript)")