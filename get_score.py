# -*- coding: utf-8 -*-
"""
stars_to_opencpop.py

Usage:
    python stars_to_opencpop.py /path/to/output.json out_opencpop.txt
"""
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple
import math

PREC = 3

# --- phoneme categories (simple chinese pinyin split) ---
CONSONANTS = set(
    [
        "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
        "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"
    ]
)
VOWELS = set(
    [
        "a", "o", "e", "i", "u", "v", "ai", "ei", "ui", "ao", "ou",
        "iu", "ie", "ve", "er", "an", "en", "ang", "eng", "ong"
    ]
)
# consider uppercase variants just in case
CONSONANTS = set([c.lower() for c in CONSONANTS])
VOWELS = set([v.lower() for v in VOWELS])

# --- utility: build intervals from labels & durations ---
def durations_to_intervals(labels: List[str], durs: List[float]) -> List[Tuple[float,float,str]]:
    intervals = []
    start = 0.0
    for lab, dur in zip(labels, durs):
        end = start + dur
        intervals.append((round(start, PREC), round(end, PREC), lab))
        start = end
    return intervals


def overlap(a0,a1,b0,b1):
    return max(0.0, min(a1,b1) - max(a0,b0))

# --- main alignment algorithm per your description ---
def align_item(item: dict):
    """
    Input item expected keys:
      item_name, word_list, word_durs, ph_list, ph_durs, note_list, note_durs
    Returns:
      tuple (id, lyrics, phonemes_list, notes_list(midi ints), syb_durs_list, ph_durs_list, slur_list)
    """
    item_name = item["item_name"]
    words = item["word_list"]
    word_durs = [float(x) for x in item["word_durs"]]
    phs = item["ph_list"]
    ph_durs_in = [float(x) for x in item["ph_durs"]]
    notes = item["note_list"]
    note_durs = [float(x) for x in item["note_durs"]]

    # build intervals
    ph_intervals = durations_to_intervals(phs, ph_durs_in)
    note_intervals = durations_to_intervals(notes, note_durs)

    # lyrics: join words except <SP> or SP
    lyrics_tokens = [w for w in words if str(w).upper() not in ("<SP>", "SP")]
    lyrics = "".join(lyrics_tokens)

    out_phs = []
    out_notes = []
    out_ph_durs = []
    out_slur = []
    out_syb_durs = []  # we will fill zeros initially

    # helper: find note intervals overlapping [L,R]
    for (p_start, p_end, p_lab) in ph_intervals:
        ph_label = str(p_lab)
        dur = round(p_end - p_start, PREC)
        low = p_start
        high = p_end
        lab_lower = ph_label.lower()
        # SP case
        if lab_lower in ("sp", "<sp>"):
            out_phs.append(ph_label)
            out_notes.append(0)
            out_ph_durs.append(round(dur, PREC))
            out_slur.append(0)
            out_syb_durs.append(0.000)
            continue

        # classify as consonant/vowel: simple check using sets (fallback: treat as vowel)
        token = lab_lower
        # sometimes phoneme labels have numbers or stress markers, strip non-letters
        token_alpha = "".join([c for c in token if c.isalpha()])
        if token_alpha in CONSONANTS:
            kind = "consonant"
        elif token_alpha in VOWELS:
            kind = "vowel"
        else:
            # fallback heuristic: single-letter vowels
            if token_alpha in ("a","e","i","o","u","v"):
                kind = "vowel"
            else:
                # if phoneme contains vowel letter -> vowel otherwise consonant
                if any(v in token_alpha for v in ["a","e","i","o","u","v"]):
                    kind = "vowel"
                else:
                    kind = "consonant"

        # collect overlapping note intervals
        overlaps = []
        for (n_s, n_e, n_lab) in note_intervals:
            ov = overlap(low, high, n_s, n_e)
            if ov > 1e-9:
                overlaps.append((n_s, n_e, n_lab, ov))

        if kind == "consonant":
            # find non-zero note with max overlap
            note_counts = {}
            for n_s, n_e, n_lab, ov in overlaps:
                # midi_val = note_name_to_midi(n_lab)
                midi_val = n_lab
                if midi_val != 0:
                    note_counts[midi_val] = note_counts.get(midi_val, 0.0) + ov
            if note_counts:
                # pick max overlap
                chosen = max(note_counts.items(), key=lambda x: x[1])[0]
            else:
                # repair later by neighbor rule: temporarily set 0
                chosen = 0
            out_phs.append(ph_label)
            out_notes.append(chosen)
            out_ph_durs.append(round(dur, PREC))
            out_slur.append(0)
            out_syb_durs.append(0.000)
        else:
            # vowel: may cover multiple note segments
            # For each overlapping note interval, create a segment
            # But zeros should be covered by previous non-zero within the vowel interval
            segments = []  # list of (midi_val, seg_dur)
            prev_nonzero = None
            # iterate note_intervals in temporal order and slice by [low, high]
            for (n_s, n_e, n_lab) in note_intervals:
                if n_e <= low or n_s >= high:
                    continue
                seg_l = max(low, n_s)
                seg_r = min(high, n_e)
                seg_len = seg_r - seg_l
                if seg_len <= 0:
                    continue
                midi_val = n_lab
                # midi_val = note_name_to_midi(n_lab)
                if midi_val == 0:
                    # zero: we'll extend previous non-zero if exists,
                    # otherwise mark as 0 (repair later)
                    if prev_nonzero is not None:
                        segments.append((prev_nonzero, seg_len))
                    else:
                        # unknown for now
                        segments.append((0, seg_len))
                else:
                    segments.append((midi_val, seg_len))
                    prev_nonzero = midi_val
            # If no overlaps found (rare), mark as 0 and repair later
            if not segments:
                segments = [(0, dur)]
            # merge consecutive same midi segments
            merged = []
            for m, l in segments:
                if merged and merged[-1][0] == m:
                    merged[-1] = (m, round(merged[-1][1] + l, 10))
                else:
                    merged.append((m, l))
            # now append one or multiple outputs: first seg slur=0, subsequent slur=1
            for idx, (midi_val, seg_len) in enumerate(merged):
                out_phs.append(ph_label)
                out_notes.append(midi_val)
                out_ph_durs.append(round(seg_len, PREC))
                out_slur.append(0 if idx == 0 else 1)
                out_syb_durs.append(0.000)

    # --- compute syllable durations ---
    out_syb_durs = [0.0] * len(out_phs)
    i = 0
    n = len(out_phs)
    while i < n:
        ph_label = str(out_phs[i]).lower()
        # 判断是否辅音
        token_alpha = "".join([c for c in ph_label if c.isalpha()])
        is_consonant = token_alpha in CONSONANTS
        # 音节起始位置
        start_idx = i
        dur_sum = 0.0
        if not is_consonant:
            # 元音，累加同音节的 ph_durs
            while i < n:
                dur_sum += out_ph_durs[i]
                # 如果下一个是同一音素延音 (slur=1)，继续累加
                if i+1 < n and out_slur[i+1] == 1 and str(out_phs[i+1]).lower() == ph_label:
                    i += 1
                else:
                    break
            end_idx = i
        else:
            # 辅音，向后找元音及延音
            found = False
            j = i
            while j < n:
                ph_j_label = str(out_phs[j]).lower()
                token_j_alpha = "".join([c for c in ph_j_label if c.isalpha()])
                if token_j_alpha not in CONSONANTS:
                    # 找到元音
                    dur_sum += out_ph_durs[j]
                    # 连续延音累加
                    k = j
                    while k+1 < n and out_slur[k+1] == 1 and str(out_phs[k+1]).lower() == ph_j_label:
                        k += 1
                        dur_sum += out_ph_durs[k]
                    end_idx = k
                    found = True
                    break
                j += 1
            if not found:
                # 没找到元音，则音节就是自己
                dur_sum = out_ph_durs[i]
                end_idx = i
        # 填充 out_syb_durs
        for idx in range(start_idx, end_idx+1):
            out_syb_durs[idx] = round(dur_sum, PREC)
        i = end_idx + 1

    # --- repair stage: any non-SP phoneme with note==0 must be fixed ---
    # we only repair non-SP entries (we used 0 for SP earlier)
    n = len(out_notes)
    for i in range(n):
        if out_notes[i] == 0 and str(out_phs[i]).lower() not in ("sp", "<sp>"):
            # find left non-zero
            left = None
            for j in range(i-1, -1, -1):
                if out_notes[j] != 0:
                    left = out_notes[j]
                    break
            right = None
            for j in range(i+1, n):
                if out_notes[j] != 0:
                    right = out_notes[j]
                    break
            if left is not None and right is not None:
                val = int(round((left + right) / 2.0))
            elif left is not None:
                val = left
            elif right is not None:
                val = right
            else:
                val = 60  # fallback central C4
            out_notes[i] = val

    # final check lengths
    assert len(out_phs) == len(out_notes) == len(out_ph_durs) == len(out_slur) == len(out_syb_durs)

    # round ph durs to PREC and ensure sum roughly equals original
    out_ph_durs = [round(x, PREC) for x in out_ph_durs]
    out_syb_durs = [round(x, PREC) for x in out_syb_durs]

    return (item_name, lyrics, out_phs, out_notes, out_syb_durs, out_ph_durs, out_slur)


def main(json_path, out_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON root to be a list of items.")

    out_lines = []
    for item in data:
        (item_name, lyrics, phs, notes, syb_durs, ph_durs, slur) = align_item(item)
        # format lists to strings
        phs_str = " ".join([str(x) for x in phs])
        notes_str = " ".join([str(int(x)) for x in notes])
        syb_str = " ".join([f"{x:.3f}" for x in syb_durs])
        phd_str = " ".join([f"{x:.3f}" for x in ph_durs])
        slur_str = " ".join([str(int(x)) for x in slur])
        line = f"{item_name}|{lyrics}|{phs_str}|{notes_str}|{syb_str}|{phd_str}|{slur_str}"
        out_lines.append(line)

    with open(out_path, "w", encoding="utf-8") as f:
        for l in out_lines:
            f.write(l + "\n")
    print(f"Saved {len(out_lines)} entries to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python stars_to_opencpop.py /path/to/output.json out_opencpop.txt")
        sys.exit(1)
    json_path = sys.argv[1]
    out_path = sys.argv[2]
    main(json_path, out_path)
