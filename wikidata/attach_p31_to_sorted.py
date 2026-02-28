import json
import os
from typing import Dict, Set, List
import sys

# --- 設定 ---
ORIGINAL_DATA_FILE = "wikidata_cl_sorted.jsonl" # 元のQIDデータファイル
P31_MAP_FILE = "qid_p31_list_simple.jsonl" # QIDとP31が対応付けられたファイル
P31_LABEL_FILE = "p31_optimized_labels.jsonl"          # ★ P31 QIDとラベルの対応ファイル ★
OUTPUT_FILE = "wikidata_with_p31_labels.jsonl"       # 出力ファイル名も変更
# -------------

def load_qid_label_map(label_path: str) -> Dict[str, str]:
    """
    QIDとラベルの対応ファイル（JSONL形式）を辞書としてメモリにロードする。
    例: {"Q5": "human"} -> {"Q5": "human"}
    """
    label_map = {}
    print(f"🔄 Loading P31 QID labels from: {label_path}")
    line_count = 0
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                try:
                    item = json.loads(line.strip())
                    # 形式が {"QID": "label"} を想定
                    qid, label = next(iter(item.items()))
                    if qid and label:
                        label_map[qid] = label
                except Exception:
                    # JSONDecodeError や next(iter) エラーをまとめて処理
                    continue
        print(f"✅ P31 Label map loaded. Total QIDs: {len(label_map)}")
    except FileNotFoundError:
        print(f"❌ Critical Error: P31 Label file not found at {label_path}", file=sys.stderr)
        raise
    return label_map

def load_p31_map(p31_map_path: str) -> Dict[str, List[str]]:
    """
    QIDとP31の対応ファイル（JSONL形式）を辞書としてメモリにロードする。
    """
    p31_map = {}
    print(f"🔄 Loading P31 map from: {p31_map_path}")
    line_count = 0
    PROGRESS_INTERVAL = 1000000  # 100万件ごとに出力
    
    try:
        with open(p31_map_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if line_count % PROGRESS_INTERVAL == 0:
                    print(f"    ... {line_count:,} lines loaded.")
                
                try:
                    item = json.loads(line.strip())
                    qid = item.get("qid")
                    p31_list = item.get("p31", [])
                    if qid and p31_list is not None:
                        p31_map[qid] = p31_list
                except json.JSONDecodeError:
                    print(f"⚠️ Warning: Skip invalid JSON line in P31 map at line {line_count:,}.", file=sys.stderr)
                    continue
                    
        print(f"✅ P31 map loaded. Total QIDs: {len(p31_map)} (Lines read: {line_count:,})")
    except FileNotFoundError:
        print(f"❌ Critical Error: P31 map file not found at {p31_map_path}", file=sys.stderr)
        raise
    return p31_map

def transform_data(original_data_path: str, p31_map: Dict[str, List[str]], p31_label_map: Dict[str, str], output_path: str):
    """
    元のデータファイルを読み込み、P31 QIDをラベルに変換して新しいファイルに出力する。
    """
    print(f"📝 Starting transformation. Writing to: {output_path}")
    
    transformed_count = 0
    PROGRESS_INTERVAL = 1000000
    
    try:
        with open(original_data_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                transformed_count += 1
                
                if transformed_count % PROGRESS_INTERVAL == 0:
                    print(f" ... {transformed_count:,} lines processed.")
                
                try:
                    original_entry = json.loads(line.strip())
                    if not original_entry: continue
                    
                    qid, original_data = next(iter(original_entry.items()))
                    
                    # 1. QIDに対応するP31 QIDリストを取得
                    p31_qids = p31_map.get(qid, [])
                    
                    # 2. ★ P31 QIDをラベルに変換 ★
                    p31_labels = []
                    for p_qid in p31_qids:
                        # マップからラベルを取得。見つからない場合はQIDをそのまま使う
                        label = p31_label_map.get(p_qid, p_qid) 
                        p31_labels.append(label)
                    
                    # 3. 新しい構造を作成
                    new_data = {
                        "label": original_data.get("label"),
                        "description": original_data.get("description"),
                        # aliasesを削除し、p31_labelsのリストを追加
                        "p31_labels": p31_labels
                    }
                    
                    transformed_entry = {qid: new_data}
                    outfile.write(json.dumps(transformed_entry, ensure_ascii=False) + '\n')
                    
                except Exception as e:
                    print(f"⚠️ Warning: Failed to process line {transformed_count:,} in original data: {line.strip()} ({e})", file=sys.stderr)
                    continue
                    
        print(f"✅ Transformation complete. Total entries processed: {transformed_count:,}")

    except FileNotFoundError:
        print(f"❌ Critical Error: Original data file not found at {original_data_path}", file=sys.stderr)


# --- 実行ブロック ---
if __name__ == "__main__":
    
    # --- デモンストレーション用のファイル作成（実際の実行時はスキップ可能） ---
    
    if not os.path.exists(ORIGINAL_DATA_FILE):
        print(f"Creating dummy file: {ORIGINAL_DATA_FILE} (for demonstration)")
        sample_data = [
            {"Q31": {"label": "Belgium", "description": "country in western Europe", "aliases": ["Kingdom of Belgium"]}},
            {"Q8": {"label": "happiness", "description": "positive emotional state", "aliases": ["happy", "gladness", "glad"]}},
            {"Q23": {"label": "George Washington", "description": "Founding Father and first U.S. president (1789–1797)", "aliases": ["Father of the United States", "American Fabius", "American Cincinnatus"]}},
            {"Q42": {"label": None, "description": "British science fiction writer and humorist (1952–2001)", "aliases": []}}
        ]
        with open(ORIGINAL_DATA_FILE, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    if not os.path.exists(P31_MAP_FILE):
        print(f"Creating dummy file: {P31_MAP_FILE} (for demonstration)")
        p31_map_data = [
            {"qid": "Q31", "p31": ["Q43702", "Q6256", "Q3624078"]},
            {"qid": "Q8", "p31": ["Q60539479", "Q331769"]},
            {"qid": "Q23", "p31": ["Q5"]},
            {"qid": "Q42", "p31": ["Q5"]},
        ]
        with open(P31_MAP_FILE, 'w', encoding='utf-8') as f:
            for item in p31_map_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # ★ P31 ラベルファイルを作成 ★
    if not os.path.exists(P31_LABEL_FILE):
        print(f"Creating dummy file: {P31_LABEL_FILE} (for demonstration)")
        p31_label_data = [
            {"Q1": "universe"},
            {"Q4": "death"},
            {"Q5": "human"},
            {"Q38": "Italy"},
            {"Q43702": "sovereign state"},  # Q31のP31に対応
            {"Q6256": "country"},           # Q31のP31に対応
            {"Q331769": "feeling"},         # Q8のP31に対応
            {"Q3624078": "member state"},   # Q31のP31に対応
            # Q60539479 のラベルはマップにない場合を想定
        ]
        with open(P31_LABEL_FILE, 'w', encoding='utf-8') as f:
            for item in p31_label_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    # ----------------------------------------------------------

    try:
        # 1. P31ラベルマップをロード
        p31_label_map_dict = load_qid_label_map(P31_LABEL_FILE)
        
        # 2. P31 QIDマップをロード
        p31_map_dict = load_p31_map(P31_MAP_FILE)
        
        # 3. データの変換と出力
        transform_data(ORIGINAL_DATA_FILE, p31_map_dict, p31_label_map_dict, OUTPUT_FILE)
        
        print(f"\n完了しました。結果は '{OUTPUT_FILE}' に出力されました。")
        
    except FileNotFoundError:
        print("\n処理を中断しました。必要な入力ファイルが存在するか確認してください。")