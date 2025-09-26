# -*- coding: utf-8 -*-
import os
import json
import textgrid

def json_to_textgrid(json_path, output_dir=None, precision=3):
    import os
    import json
    import textgrid

    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    if not isinstance(data_list, list):
        raise ValueError("JSON root must be a list of items.")

    for item in data_list:
        item_name = item["item_name"]

        ph_list = item["ph_list"]
        ph_durs = item["ph_durs"]
        word_list = item["word_list"]
        word_durs = item["word_durs"]
        note_list = item["note_list"]
        note_durs = item["note_durs"]

        # 累加 duration 生成 start/end 时间（全精度）
        def durations_to_intervals(labels, durs):
            intervals = []
            start = 0.0
            for label, dur in zip(labels, durs):
                end = start + dur
                intervals.append((start, end, label))
                start = end
            return intervals

        word_intervals = durations_to_intervals(word_list, word_durs)
        ph_intervals = durations_to_intervals(ph_list, ph_durs)
        note_intervals = durations_to_intervals(note_list, note_durs)

        # TextGrid 对象
        tg = textgrid.TextGrid()

        # Word Tier
        word_max = round(sum(word_durs), precision)
        word_tier = textgrid.IntervalTier(name="words", minTime=0.0, maxTime=word_max)
        for start, end, label in word_intervals:
            word_tier.add(round(start, precision), round(end, precision), label)
        tg.append(word_tier)

        # Phone Tier
        ph_max = round(sum(ph_durs), precision)
        ph_tier = textgrid.IntervalTier(name="phones", minTime=0.0, maxTime=ph_max)
        for start, end, label in ph_intervals:
            ph_tier.add(round(start, precision), round(end, precision), label)
        tg.append(ph_tier)

        # Note Tier
        note_max = round(sum(note_durs), precision)
        note_tier = textgrid.IntervalTier(name="note", minTime=0.0, maxTime=note_max)
        for start, end, label in note_intervals:
            note_tier.add(round(start, precision), round(end, precision), str(label))
        tg.append(note_tier)

        # 输出路径
        if output_dir is None:
            output_dir = os.path.dirname(json_path)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{item_name}.TextGrid")
        tg.write(output_path)
        print(f"Saved TextGrid for {item_name} to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, default=None, help="Output folder for TextGrid")
    args = parser.parse_args()

    json_to_textgrid(args.json, args.output)
