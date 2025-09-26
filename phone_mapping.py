def is_initial(ph):
    """判断是否是声母"""
    initials = {
        "b", "p", "m", "f",
        "d", "t", "n", "l",
        "g", "k", "h",
        "j", "q", "x",
        "zh", "ch", "sh", "r",
        "z", "c", "s",
        "y", "w"
    }
    return ph in initials


def load_mapping(mapping_file):
    """读入 gts2open.txt: 1 -> 2 phoneme"""
    mapping = {}
    with open(mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            k, v = line.split("|")
            mapping[k] = v.split()  # 例如 "iou" -> ["y", "ou"]
    return mapping


def replace_vowels_with_mapping(input_file, output_file, mapping_file):
    """同时实现:
       1个ph → 2个ph
       1个ph → 1个ph (替换韵母)
    """

    # 一对一替换表
    # 非法字符表
    simple_mapping = {
        "iou": "iu",
        "uei": "ui",
        "uen": "un"
    }

    # 一对二替换表
    split_mapping = load_mapping(mapping_file)

    total_lines, success_lines = 0, 0
    replaced_count = 0

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            total_lines += 1
            parts = line.strip().split("|")
            if len(parts) != 7:
                f_out.write(line)
                continue

            pid, lyrics, phs_str, notes_str, syb_durs_str, ph_durs_str, slurs_str = parts

            phs = phs_str.split()
            notes = notes_str.split()
            syb_durs = syb_durs_str.split()
            ph_durs = ph_durs_str.split()
            slurs = slurs_str.split()

            new_phs, new_notes, new_syb_durs, new_ph_durs, new_slurs = [], [], [], [], []

            i = 0
            while i < len(phs):
                ph = phs[i]

                # ---- 功能 1: 一对二替换 (孤立韵母) ----
                if ph in split_mapping:
                    # if i == 0 or not is_initial(phs[i - 1]):
                    if (i == 0 or not is_initial(phs[i - 1])) and slurs[i] != "1":
                        mapped = split_mapping[ph]  # [y, ou] 等
                        note, syb, slur = notes[i], syb_durs[i], slurs[i]
                        dur = float(ph_durs[i])

                        dur1, dur2 = round(dur * 0.3, 4), round(dur * 0.7, 4)

                        new_phs.extend(mapped)
                        new_notes.extend([note, note])
                        new_syb_durs.extend([syb, syb])
                        new_ph_durs.extend([str(dur1), str(dur2)])
                        new_slurs.extend([slur, slur])
                        replaced_count += 1
                        i += 1
                        continue

                # ---- 功能 2: 一对一替换 (前面是声母) ----
                if ph in simple_mapping and i > 0:
                    ph = simple_mapping[ph]
                    replaced_count += 1
                    # if is_initial(phs[i - 1]):
                    #     ph = simple_mapping[ph]
                    #     replaced_count += 1

                # ---- 默认复制 ----
                new_phs.append(ph)
                new_notes.append(notes[i])
                new_syb_durs.append(syb_durs[i])
                new_ph_durs.append(ph_durs[i])
                new_slurs.append(slurs[i])
                i += 1
            
            # <<< 在这里加补丁 >>>
            # 修正加入延音引起的错误
            for j in range(1, len(new_phs)):
                if new_slurs[j] == "1":
                    new_phs[j] = new_phs[j - 1]

            # 检查对齐
            if len(new_phs) == len(new_notes) == len(new_syb_durs) == len(new_ph_durs) == len(new_slurs):
                success_lines += 1
            else:
                print(f"[Warning] 对齐失败: {pid}")

            f_out.write(
                "|".join([
                    pid, lyrics,
                    " ".join(new_phs),
                    " ".join(new_notes),
                    " ".join(new_syb_durs),
                    " ".join(new_ph_durs),
                    " ".join(new_slurs),
                ]) + "\n"
            )

    print(f"处理完成: {success_lines}/{total_lines} 行成功 ({success_lines/total_lines:.2%})")
    print(f"替换次数: {replaced_count}")


def check_illegal_vowels(file_path):
    """检查是否还有非法韵母 iou/uei/uen"""
    illegal = {"iou", "uei", "uen"}
    bad_lines = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            phs = parts[2].split()
            found = [ph for ph in phs if ph in illegal]
            if found:
                bad_lines.append((idx, found))

    if bad_lines:
        print("发现非法韵母：")
        for ln, fnd in bad_lines:
            print(f"  行 {ln}: {', '.join(fnd)}")
    else:
        print("没有发现非法韵母。")

if __name__ == "__main__":
    score_file = "out_opencpop.txt"
    output_file = score_file.replace(".txt", "_processed.txt")
    replace_vowels_with_mapping(
        score_file,
        output_file,
        "gts2open.txt"
    )

    check_illegal_vowels(output_file)
