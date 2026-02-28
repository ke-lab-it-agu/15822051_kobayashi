import bz2
import json
import re
import time
from collections import defaultdict

# --- 設定 ---
input_file = "latest-truthy.nt.bz2"
output_file = "qid_p31_list_simple.jsonl" 
# キャッシュサイズによる書き出しトリガー (参照コードに合わせ50,000件)
FLUSH_CACHE_SIZE = 50_000 

# P31 (instance of) トリプルの正規表現
# 主語、述語、目的語の間にスペース (' ') が必要です。
# グループ1: 主語 (Subject) のQID URI | グループ2: 目的語 (Object) のQID URI
p31_pattern = re.compile(
    r'<(http://www.wikidata.org/entity/Q\d+)> '      # Subject URI とスペース
    r'<http://www.wikidata.org/prop/direct/P31> '    # Predicate URI とスペース
    r'<(http://www.wikidata.org/entity/Q\d+)>'       # Object URI (QID)
)

# QIDごとにP31 IDのセットを保持
# temp_cache[QID_URI] = {P31_URI_A, P31_URI_B, ...}
temp_cache = defaultdict(set) 
written_qids = set() # ファイルに書き込み済みのユニークQIDを記録

start = time.time()
processed = 0      # 処理されたP31トリプルの総数
processed_lines = 0 # ファイルの総行数（進捗ログ用）

def flush_to_file(cache, f):
    """キャッシュの内容をファイルに書き出し、キャッシュをクリアする"""
    global written_qids
    
    written_count = 0
    for qid_uri, p31_set in cache.items():
        # URIから純粋なQID (例: Q123) のみを取り出す
        qid = qid_uri.split('/')[-1]
        # P31 URIから純粋なQID (例: Q42) のみを取り出す
        p31_list = [p.split('/')[-1] for p in p31_set]

        # 書き出し形式: {"qid": "Q123", "p31": ["Q42", "Q5", ...]}
        json.dump({"qid": qid, "p31": p31_list}, f, ensure_ascii=False)
        f.write("\n")
        
        # 書き込み済みQIDを記録 (URIではなく純粋なQIDを記録)
        written_qids.add(qid)
        written_count += 1

    cache.clear()
    return written_count

print(f"--- Wikidata P31 (instance of) 抽出 (逐次書き出しモード) ---")
print(f"入力: {input_file}")
print(f"出力: {output_file}\n")

try:
    with bz2.open(input_file, "rt", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        for line in fin:
            processed_lines += 1
            
            # P31トリプルを正規表現で高速に抽出
            match = p31_pattern.search(line)
            if match:
                # 主語のURIと目的語のURIを抽出
                subject_qid_uri = match.group(1)
                object_p31_uri = match.group(2)

                temp_cache[subject_qid_uri].add(object_p31_uri)
                processed += 1 # P31トリプル数をカウント

            # --- 進捗ログ (ファイルの総行数ベース) ---
            if processed_lines % 1_000_000 == 0:
                elapsed = time.time() - start
                print(
                    f"[PROGRESS] {processed_lines:,} lines | "
                    f"P31 triples: {processed:,} | "
                    f"buffer: {len(temp_cache):,} | "
                    f"written QIDs: {len(written_qids):,}"
                )

            # --- キャッシュ書き出し条件 (バッファサイズベース) ---
            if len(temp_cache) > FLUSH_CACHE_SIZE:
                written_count = flush_to_file(temp_cache, fout)
                print(f"[FLUSH] Wrote {written_count:,} entities. Total written QIDs: {len(written_qids):,}")

        # --- 最終 flush (残ったバッファを書き出す) ---
        if temp_cache:
            written_count = flush_to_file(temp_cache, fout)
            print(f"\n[FINAL FLUSH] Wrote {written_count:,} entities. Total written QIDs: {len(written_qids):,}")


except FileNotFoundError:
    print(f"\n[エラー] ファイルが見つかりません: {input_file}")
    print("ファイルをダウンロードするか、正しいファイルパスに変更してください。")
except Exception as e:
    print(f"\n[重大なエラー] 処理中に例外が発生しました: {e}")

# 最終結果の表示
end_time = time.time()
print(f"\n--- 処理結果 ---")
print(f"✅ 完了: {output_file}")
print(f"総処理行数: {processed_lines:,} 行")
print(f"総 P31 トリプル数: {processed:,} 件")
print(f"処理時間: {end_time - start:.2f} 秒")
print(f"出力されたユニークな QID 総数: {len(written_qids):,} 件")