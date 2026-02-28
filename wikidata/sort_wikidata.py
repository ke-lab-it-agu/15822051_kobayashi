import heapq
import json

input_file = "max_cleaned_wikidata.jsonl"
output_file = "wikidata_cl_sorted.jsonl"
chunk_size = 10_000_000  # 1回に読み込むQID数

def qid_key(qid):
    """下4桁無視して整数で比較"""
    num = int(qid[1:])  # Q12345678 -> 12345678
    return num // 10_000  # 下4桁切り捨て

# ---------------------------
# 1. チャンクに分けて読み込み、メモリ内でソートして一時ファイルに書く
# ---------------------------
chunk_files = []
chunk = []
count = 0
chunk_index = 0

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        qid = next(iter(obj))
        chunk.append(line)
        count += 1

        if count >= chunk_size:
            chunk.sort(key=lambda x: qid_key(next(iter(json.loads(x)))))
            chunk_file = f"tmp_chunk_{chunk_index}.jsonl"
            with open(chunk_file, "w", encoding="utf-8") as cf:
                cf.writelines(chunk)
            chunk_files.append(chunk_file)
            print(f"[CHUNK] {chunk_index} written, {count:,} lines")
            chunk.clear()
            count = 0
            chunk_index += 1

# 残りのチャンク
if chunk:
    chunk.sort(key=lambda x: qid_key(next(iter(json.loads(x)))))
    chunk_file = f"tmp_chunk_{chunk_index}.jsonl"
    with open(chunk_file, "w", encoding="utf-8") as cf:
        cf.writelines(chunk)
    chunk_files.append(chunk_file)
    print(f"[CHUNK] {chunk_index} written, {len(chunk):,} lines")

# ---------------------------
# 2. 複数チャンクをマージソート
# ---------------------------
def gen_qid_key_line(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            qid = next(iter(json.loads(line)))
            yield (qid_key(qid), line)

generators = [gen_qid_key_line(f) for f in chunk_files]
total_lines = 0

with open(output_file, "w", encoding="utf-8") as out:
    for _, line in heapq.merge(*generators):
        out.write(line)
        total_lines += 1
        if total_lines % 1_000_000 == 0:
            print(f"[MERGE] {total_lines:,} lines written")

print(f"完了: {output_file}, 合計 {total_lines:,} 行")
