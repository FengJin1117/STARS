import os
import json
import sys

def read_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(list_of_dict, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list_of_dict, f, ensure_ascii=False, indent=2)

def get_syllable_durations(ph_list, ph_durs, ph2words):
    """
    根据 phoneme 序列、phoneme 时长、ph2words 映射，
    生成与 ph_list 等长的 syllable durations。

    规则：
    - <SP> 不参与 syllable，直接复制 ph_dur
    - 非 <SP> phoneme：按 ph2words 分组，syb_dur = 该 word 下所有 ph_dur 之和
    """

    if len(ph_list) != len(ph_durs):
        raise ValueError(
            f"ph_list and ph_durs must have the same length, "
            f"got {len(ph_list)} vs {len(ph_durs)}"
        )

    # === 1. 统计每个 word_id 的总 phoneme 时长 ===
    word_dur_map = {}   # word_id -> total duration
    w_idx = 0           # index for ph2words

    for ph, dur in zip(ph_list, ph_durs):
        if str(ph).upper() in ("<SP>", "SP"):
            continue

        if w_idx >= len(ph2words):
            raise ValueError(
                "ph2words length is shorter than non-SP phonemes"
            )

        word_id = ph2words[w_idx]
        word_dur_map[word_id] = word_dur_map.get(word_id, 0.0) + float(dur)
        w_idx += 1

    if w_idx != len(ph2words):
        raise ValueError(
            f"Unused ph2words entries: used {w_idx}, total {len(ph2words)}"
        )

    # === 2. 回填 syllable durations（与 ph_list 等长） ===
    syb_durs = []
    w_idx = 0

    for ph, dur in zip(ph_list, ph_durs):
        if str(ph).upper() in ("<SP>", "SP"):
            syb_durs.append(float(dur))
        else:
            word_id = ph2words[w_idx]
            syb_durs.append(word_dur_map[word_id])
            w_idx += 1

    return syb_durs

def fix_notes(note_list, ph_list):
    if len(note_list) != len(ph_list):
        raise ValueError(
            f"note_list and ph_list must have the same length, "
            f"got {len(note_list)} vs {len(ph_list)}"
        )
    # --- repair stage: any non-SP phoneme with note==0 must be fixed ---
    # we only repair non-SP entries (we used 0 for SP earlier)
    n = len(note_list)
    for i in range(n):
        if note_list[i] == 0 and str(ph_list[i]).lower() not in ("sp", "<sp>"):
            # find left non-zero
            left = None
            for j in range(i-1, -1, -1):
                if note_list[j] != 0:
                    left = note_list[j]
                    break
            right = None
            for j in range(i+1, n):
                if note_list[j] != 0:
                    right = note_list[j]
                    break
            # 就近原则取note，为了让旋律更平滑
            if left is not None and right is not None: # 左右平均
                val = int(round((left + right) / 2.0))
            elif left is not None:
                val = left
            elif right is not None:
                val = right
            else:
                val = 60  # fallback central C4
            note_list[i] = val
            
    return note_list

def align_notes(ph_list, ph_durs, note_list, note_durs):
    '''
        把note_list对齐到ph_list。得到每个ph_list的音高。
        根据两条时间轴对齐：ph_durs、note_durs。
        核心规则：
        - 如果ph是<SP>，那么note为0；
        - 如果不是，那么需要在ph的区间[a, b]，对里面覆盖住的note，加权平均，四舍五入
    '''
    if len(note_list) != len(note_durs):
        raise ValueError(
            f"note_list and note_durs must have same length, "
            f"got {len(note_list)} vs {len(note_durs)}"
        )

    if len(ph_list) != len(ph_durs):
        raise ValueError(
            f"ph_list and ph_durs must have same length, "
            f"got {len(ph_list)} vs {len(ph_durs)}"
        )

    # === 1. build time intervals ===
    ph_intervals = []
    t = 0.0
    for d in ph_durs:
        ph_intervals.append((t, t + d))
        t += d

    note_intervals = []
    t = 0.0
    for d in note_durs:
        note_intervals.append((t, t + d))
        t += d

    # === 2. 【对齐】align with duration-weighted pitch ===
    aligned_note_list = []

    for i, ph in enumerate(ph_list):
        if str(ph).lower() in ("sp", "<sp>"):
            aligned_note_list.append(0)
            continue

        ph_start, ph_end = ph_intervals[i]

        weighted_sum = 0.0
        total_overlap = 0.0

        for note, (n_start, n_end) in zip(note_list, note_intervals):
            if note == 0:
                continue

            overlap = min(ph_end, n_end) - max(ph_start, n_start)
            if overlap > 0:
                weighted_sum += overlap * note
                total_overlap += overlap

        if total_overlap > 0:
            aligned_note = int(round(weighted_sum / total_overlap))
        else:
            aligned_note = 0  # 极端情况，交给 fix_notes 兜底

        aligned_note_list.append(aligned_note)

    # === 3.【补丁】修复bug：如果ph不是<SP>，但是note是0，处理非法情况 ===
    aligned_note_list = fix_notes(aligned_note_list, ph_list)

    return aligned_note_list


def convert_item_to_gtsinger(item, ph2words_map):
    """
    将单条 STARS output item 转换为 GTSinger 格式的乐谱 dict。

    约定：
    - 不删除原有字段
    - 只在 item 基础上新增字段
    - 返回一个新的 dict（不原地修改）
    """

    item_name = item["item_name"]

    # 深拷贝，避免污染原始数据
    score = dict(item)

    # === 1. Get Syllables Durations ===
    ph_list = score["ph_list"]
    ph_durs = score["ph_durs"]
    ph2words = ph2words_map.get(item_name)
    if ph2words is None:
        raise KeyError(f"metadata.json 中未找到 item_name: {item_name}")

    syb_durs = get_syllable_durations(ph_list, ph_durs, ph2words)
    score["syb_durs"] = syb_durs

    # === 2. 把note_list对齐到ph_list ===
    note_list = score["note_list"]
    note_durs = score["note_durs"]
    aligned_note_list = align_notes(ph_list, ph_durs, note_list, note_durs)
    score["note_list"] = aligned_note_list # 替换成对齐后的note_list

    # === 3. 添加：GTSinger 特有字段 ===
    note_type = [1 if ph=="SP" else 2 for ph in ph_list]
    score["ep_types"] = note_type

    # === 4. 校验（后续补） ===
    # TODO: assert 时长守恒 / 长度一致

    return score


def convert_item_to_opencpop(data):
    '''
        return: 返回一个字符串
    '''
    for item in data:
        # 读入数据
        item_name = data["item_name"]
        word_list = data["word_list"]
        ph_list = data["ph_list"]
        note_list = data["note_list"]
        syb_durs = data["syb_durs"]
        ph_durs = data["ph_durs"]

        # format lists to strings
        lyrics = "".join([str(x) for x in word_list if x != "<SP>"])
        phs_str = " ".join([str(x) for x in ph_list])
        notes_str = " ".join([str(int(x)) for x in note_list])
        syb_str = " ".join([f"{x:.3f}" for x in syb_durs])
        phd_str = " ".join([f"{x:.3f}" for x in ph_durs])
        slur_str = ["0"] * len(ph_list)
        line = f"{item_name}|{lyrics}|{phs_str}|{notes_str}|{syb_str}|{phd_str}|{slur_str}"
    return line

def gtsinger_to_opencpop(gtsinger_path):
    '''
        这里负责遍历数据，读入保存。不负责具体转换功能
    '''
    if os.path.exists(gtsinger_path):
        raise FileNotFoundError(
            f"文件不存在：{gtsinger_path}"
        )
    data = read_json(gtsinger_path)

    opencpop_lines = []
    for item in data:
        score_line = convert_item_to_gtsinger(item)
        opencpop_lines.append(score_line)
    
    # 输出opencpop乐谱
    opencpop_path = gtsinger_path.replace("output.json", "opencpop.txt")
    with open(opencpop_path, "w", encoding="utf-8") as f:
        for l in opencpop_lines:
            f.write(l + "\n")
    print(f"保存 {len(opencpop_lines)} 行到 to {opencpop_path}")

def stars_to_gtsinger(json_path):
    """
    根据 STARS 的 output.json，生成 GTSinger 格式乐谱（json）。
    这里只负责遍历每个item，不管具体处理。
    """

    # 1. 读取 output.json
    data = read_json(json_path)

    assert isinstance(data, list), "output.json 应为 list[dict]"

    # 2. 读取 metadata.json（后续用 ph2words / 原始信息）
    metadata_path = json_path.replace("output.json", "metadata.json")
    metadata = read_json(metadata_path)

    # 建立 item_name -> ph2words 映射
    ph2words_map = {item["item_name"]: item["ph2words"] for item in metadata}

    # 3. 转换每一个 item
    gtsinger_scores = []
    for item in data:
        score_dict = convert_item_to_gtsinger(item, ph2words_map)
        gtsinger_scores.append(score_dict)

    # 4. 保存输出gtsinger形式乐谱
    gtsinger_path = json_path.replace("output.json", "gtsinger.json")
    write_json(gtsinger_scores, gtsinger_path)
    print(f"GTSinger 格式乐谱输出到: {gtsinger_path}")

    # 5. 处理成opencpop形式乐谱
    gtsinger_to_opencpop(gtsinger_path)

# 测试函数
def test():
    json_path = "rock_new/output.json"

    stars_to_gtsinger(json_path)

# 指令模式
def command():
    json_path = sys.argv[1]
    stars_to_gtsinger(json_path)



if __name__ == "__main__":
    test()