# -*- coding: utf-8 -*-
import os
import json
import argparse
import textgrid

############################
# Part 1: JSON -> TextGrid #
############################
def json_to_textgrids(json_path, output_dir="textgrid", precision=3):
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    if not isinstance(data_list, list):
        raise ValueError("JSON root must be a list of items.")

    os.makedirs(output_dir, exist_ok=True)
    tg_paths = []

    def durations_to_intervals(labels, durs):
        intervals = []
        start = 0.0
        for label, dur in zip(labels, durs):
            end = start + dur
            intervals.append((start, end, label))
            start = end
        return intervals

    for item in data_list:
        item_name = item["item_name"]

        word_intervals = durations_to_intervals(item["word_list"], item["word_durs"])
        ph_intervals = durations_to_intervals(item["ph_list"], item["ph_durs"])
        note_intervals = durations_to_intervals(item["note_list"], item["note_durs"])

        tg = textgrid.TextGrid()

        # Word Tier
        word_max = round(sum(item["word_durs"]), precision)
        word_tier = textgrid.IntervalTier(name="words", minTime=0.0, maxTime=word_max)
        for start, end, label in word_intervals:
            word_tier.add(round(start, precision), round(end, precision), label)
        tg.append(word_tier)

        # Phone Tier
        ph_max = round(sum(item["ph_durs"]), precision)
        ph_tier = textgrid.IntervalTier(name="phones", minTime=0.0, maxTime=ph_max)
        for start, end, label in ph_intervals:
            ph_tier.add(round(start, precision), round(end, precision), label)
        tg.append(ph_tier)

        # Note Tier
        note_max = round(sum(item["note_durs"]), precision)
        note_tier = textgrid.IntervalTier(name="note", minTime=0.0, maxTime=note_max)
        for start, end, label in note_intervals:
            note_tier.add(round(start, precision), round(end, precision), str(label))
        tg.append(note_tier)

        tg_path = os.path.join(output_dir, f"{item_name}.TextGrid")
        tg.write(tg_path)
        tg_paths.append(tg_path)
        print(f"[json2tg] Saved {tg_path}")

    return tg_paths


#################################
# Part 2: TextGrid -> JSON Score #
#################################
def tg2text(tg_path):
    tg = textgrid.TextGrid.fromFile(tg_path)
    return tg[0], tg[1], tg[2]


def tg2json(tg_path, output_dir="score_json", language="chinese", precision=3):
    os.makedirs(output_dir, exist_ok=True)
    word_intervals, ph_intervals, note_intervals = tg2text(tg_path)

    word_dict_list = []
    word_id = -1
    for w in word_intervals:
        mark = w.mark.strip() if w.mark.strip() else "_NONE"
        word_dict = {
            "word": mark,
            "start_time": round(float(w.minTime), precision),
            "end_time": round(float(w.maxTime), precision),
            "word_id": word_id + 1 if mark not in ["breathe", "_NONE"] else word_id,
            "ph": [],
            "ph_start": [],
            "ph_end": [],
            "note": [],
            "note_start": [],
            "note_end": []
        }
        if mark not in ["breathe", "_NONE"]:
            word_id += 1
        word_dict_list.append(word_dict)

    # Phones
    idx = 0
    for ph in ph_intervals:
        mark = ph.mark.strip() if ph.mark.strip() else "_NONE"
        if mark in ["breathe", "_NONE"]:
            word_dict_list[idx]["ph"].append(mark)
            word_dict_list[idx]["ph_start"].append(word_dict_list[idx]["start_time"])
            word_dict_list[idx]["ph_end"].append(word_dict_list[idx]["end_time"])
            idx += 1
        else:
            word_dict_list[idx]["ph"].append(mark)
            word_dict_list[idx]["ph_start"].append(round(float(ph.minTime), precision))
            word_dict_list[idx]["ph_end"].append(round(float(ph.maxTime), precision))
            if float(ph.maxTime) >= word_dict_list[idx]["end_time"]:
                idx += 1

    # Notes
    idx = 0
    for note in note_intervals:
        mark = note.mark.strip()
        mark = int(mark) if mark.isdigit() else 0
        min_time, max_time = round(float(note.minTime), precision), round(float(note.maxTime), precision)
        word_dict_list[idx]["note"].append(mark)
        word_dict_list[idx]["note_start"].append(min_time)
        word_dict_list[idx]["note_end"].append(max_time)
        if max_time >= word_dict_list[idx]["end_time"]:
            idx += 1

    output_path = os.path.join(output_dir, os.path.basename(tg_path).replace('.TextGrid', '.json'))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(word_dict_list, f, ensure_ascii=False, indent=4)
    print(f"[tg2json] Saved {output_path}")
    return output_path


############################
# Main Pipeline
############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="STARS output JSON file")
    # parser.add_argument("--tg_dir", type=str, default="textgrid", help="Folder to save TextGrid")
    # parser.add_argument("--score_dir", type=str, default="score_json", help="Folder to save score JSON")
    parser.add_argument("--precision", type=int, default=3, help="Decimal precision for times")
    parser.add_argument("--language", type=str, default="chinese", help="Language tag")
    args = parser.parse_args()

    # Step 1: json -> multiple TextGrid
    tg_dir = os.path.join(os.path.dirname(args.json_path), "textgrids") # Folder to save TextGrid
    tg_paths = json_to_textgrids(args.json_path, tg_dir, precision=args.precision)

    # Step 2: batch TextGrid -> score JSON
    score_dir = os.path.join(os.path.dirname(args.json_path), "scores") # Folder to save score JSON
    for tg_path in tg_paths:
        tg2json(tg_path, output_dir=score_dir, language=args.language, precision=args.precision)
