import json
import time

input_file = "wikidata_en_qls.jsonl"
output_file = "max_cleaned_wikidata.jsonl"

# 除外キーワード（lowercase に揃えて高速化）
exclude_keywords = [
    "wikimedia disambiguation page",
    "wikimedia disambiguation category",
    "wikimedia category",
    "wikimedia template",
    "wikimedia list article",
    "wikimedia list",
    "wikimedia set",
    "wikimedia project page",
    "wikimedia module",
    "wikimedia portal",
    "wikimedia artist discography",
    "wikimedia music-related list",
    "wikimedia navigational template",
    "wikimedia duplicated page",
]

start = time.time()
processed = 0
kept = 0
skipped = 0

LOG_INTERVAL = 1_000_000  # 100万行ごとにログ

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:
        processed += 1
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)
        qid = next(iter(data))
        entry = data[qid]

        description = entry.get("description", "").lower()

        # 除外判定
        if any(key in description for key in exclude_keywords):
            skipped += 1
        else:
            fout.write(json.dumps({qid: entry}, ensure_ascii=False) + "\n")
            kept += 1

        # ログ出力（100万行ごと）
        if processed % LOG_INTERVAL == 0:
            elapsed = time.time() - start
            print(
                f"[PROGRESS] {processed:,} lines processed | "
                f"kept: {kept:,} | skipped: {skipped:,} | "
                f"time: {elapsed:.1f}s"
            )

# 最終ログ
elapsed = time.time() - start
print("\n=== 完了 ===")
print(f"総処理行数: {processed:,}")
print(f"採用: {kept:,}")
print(f"除外: {skipped:,}")
print(f"処理時間: {elapsed:.1f} 秒")
