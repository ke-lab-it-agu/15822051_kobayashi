import bz2
import json
import time

input_file = "latest-truthy.nt.bz2"
output_file = "wikidata_en_qls.jsonl"
temp_cache = {}
written_qids = set()  # 書き込んだユニークQIDを記録

start = time.time()
processed = 0
qid_counter = 0  # ラベル/説明/エイリアス総数

def flush_to_file(cache, f):
    global written_qids
    for qid, data in cache.items():
        json.dump({qid: data}, f, ensure_ascii=False)
        f.write("\n")
        written_qids.add(qid)
    cache.clear()

with bz2.open(input_file, "rt", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out:
    for line in f:
        processed += 1

        if not line.startswith("<http://www.wikidata.org/entity/Q"):
            continue

        # QID 抽出
        qid_start = line.find("Q", 30)
        if qid_start == -1:
            continue
        qid_end = line.find(">", qid_start)
        qid = line[qid_start:qid_end]

        # テキスト抽出
        first_quote = line.find('"')
        if first_quote == -1:
            continue
        second_quote = line.find('"', first_quote + 1)
        if second_quote == -1:
            continue
        text = line[first_quote + 1:second_quote]

        is_en = "@en" in line
        is_label = "<http://www.w3.org/2000/01/rdf-schema#label>" in line
        is_desc = "<http://schema.org/description>" in line
        is_alias = "<http://www.w3.org/2004/02/skos/core#altLabel>" in line

        # ラベル：英語のみ
        if is_label and is_en:
            temp_cache.setdefault(qid, {})["label"] = text
            qid_counter += 1

        # 説明：英語のみ
        elif is_desc and is_en:
            temp_cache.setdefault(qid, {})["description"] = text
            qid_counter += 1

        # エイリアス：英語のみ
        elif is_alias and is_en:
            temp_cache.setdefault(qid, {}).setdefault("aliases", []).append(text)
            qid_counter += 1

        # 進捗ログ（書き込まれたユニークQID数を表示）
        if processed % 1_000_000 == 0:
            print(
                f"[PROGRESS] {processed:,} lines | "
                f"buffer: {len(temp_cache):,} | "
                f"QIDs entries: {qid_counter:,} | "
                f"written QIDs: {len(written_qids):,}"
            )

        if len(temp_cache) > 50_000:
            flush_to_file(temp_cache, out)
            print(f"[FLUSH] {len(written_qids):,} unique QIDs written so far")

    # 最終 flush
    if temp_cache:
        flush_to_file(temp_cache, out)

print(f"\n✅ 完了: {output_file}")
print(f"処理時間: {time.time() - start:.2f} 秒")
print(f"総 QID 数: {len(written_qids):,}")
print(f"ラベル/説明/エイリアスの総数: {qid_counter:,}")
