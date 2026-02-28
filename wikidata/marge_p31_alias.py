import json
import os
import time
from typing import Dict, Any

# --- 設定 ---
# 実際のファイル名に合わせて変更してください
P31_INPUT_FILE = "wikidata_with_p31_labels.jsonl"    # p31_labelsを含む入力ファイル名
ALIASES_INPUT_FILE = "wikidata_cl_sorted.jsonl" # aliases情報を含む入力ファイル名
OUTPUT_FILE = "wikidata_alias_p31_m.jsonl" # 最終的な出力ファイル名

# --- ユーザー提供のサンプルデータ (デモ用) ---
# 実際には、以下のデータをファイルに保存して実行してください。
# このスクリプトは、ファイルが存在しない場合、以下のサンプルデータを使用してデモ実行します。
SAMPLE_P31_DATA = [
    {"Q1": {"label": "universe", "description": "totality consisting of space, time, matter and energy", "p31_labels": ["universe"]}},
    {"Q15": {"label": "Africa", "description": "continent", "p31_labels": ["continent", "geographic location", "geographical region"]}},
    {"Q17": {"label": "Japan", "description": "island nation in East Asia", "p31_labels": ["island nation", "country", "sovereign state"]}},
]

SAMPLE_ALIASES_DATA = [
    {"Q1": {"label": "universe", "description": "totality consisting of space, time, matter and energy", "aliases": ["macrocosm", "the cosmos"]}},
    {"Q15": {"label": "Africa", "description": "continent", "aliases": ["African continent"]}},
    {"Q17": {"label": "Japan", "description": "island nation in East Asia", "aliases": ["State of Japan", "Nihon", "Nippon"]}},
]
# ---------------------------------------------

def load_data(filepath: str, key_to_extract: str) -> Dict[str, Dict[str, Any]]:
    """
    JSONLファイルを読み込み、QIDをキーとする辞書を構築します。
    ファイルが存在しない場合は、デモ用のサンプルデータを使用します。
    """
    data_dict = {}
    total_lines = 0
    start_time = time.time()

    print(f"[{time.strftime('%H:%M:%S')}] ファイル '{filepath}' を読み込み中...")

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                try:
                    item = json.loads(line.strip())
                    # JSONLの各行は {"QID": {properties...}} の形式を想定
                    qid = list(item.keys())[0]
                    # QIDをキーとして、そのプロパティ辞書を格納
                    data_dict[qid] = item[qid]
                except (json.JSONDecodeError, IndexError, TypeError) as e:
                    print(f"[{time.strftime('%H:%M:%S')}] [警告] {filepath} の {total_lines} 行目でパースエラー: {e}. スキップします。")
                    continue
        print(f"[{time.strftime('%H:%M:%S')}] {filepath} の読み込み完了。合計 {total_lines} 行、 {len(data_dict)} 件のQIDをロードしました。")
    else:
        # デモモード: ファイルが存在しない場合はサンプルを使用
        print(f"[{time.strftime('%H:%M:%S')}] [デモモード] ファイル '{filepath}' が見つかりません。埋め込みのサンプルデータを使用します。")
        sample = SAMPLE_P31_DATA if key_to_extract == 'p31_labels' else SAMPLE_ALIASES_DATA
        for item in sample:
            qid = list(item.keys())[0]
            data_dict[qid] = item[qid]
        print(f"[{time.strftime('%H:%M:%S')}] [デモ] {len(data_dict)} 件のQIDをロードしました。")

    return data_dict

def merge_data(p31_data: Dict[str, Dict[str, Any]], aliases_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    p31_dataをベースに、aliases_dataからaliasesの情報を追記してマージします。
    """
    merge_count = 0
    total_p31_qids = len(p31_data)
    
    print(f"[{time.strftime('%H:%M:%S')}] データマージ処理を開始します (ベースQID数: {total_p31_qids})。")

    # p31_dataのQIDを順に処理し、aliasesを追記
    for qid in p31_data.keys():
        if qid in aliases_data:
            # aliasesデータに対応するQIDが存在し、aliasesプロパティがある場合
            aliases_info = aliases_data[qid].get('aliases')
            
            if aliases_info is not None:
                p31_data[qid]['aliases'] = aliases_info
                merge_count += 1
    
    print(f"[{time.strftime('%H:%M:%S')}] データマージ完了。合計 {merge_count} 件のQIDにaliases情報が追記されました。")
    return p31_data

def save_data(data_dict: Dict[str, Dict[str, Any]], filepath: str):
    """
    マージされた辞書の内容を、QIDごとに独立したJSONオブジェクトとしてJSONL形式で保存します。
    """
    total_saved = 0
    
    print(f"[{time.strftime('%H:%M:%S')}] マージ結果を '{filepath}' に保存中...")

    with open(filepath, "w", encoding="utf-8") as f:
        # 辞書内のすべてのQIDを処理
        for qid, properties in data_dict.items():
            # {"QID": {properties...}} の形式に再構築
            output_item = {qid: properties}
            # JSONL形式で書き出し (ensure_ascii=Falseで日本語など非ASCII文字を保持)
            json_line = json.dumps(output_item, ensure_ascii=False)
            f.write(json_line + "\n")
            total_saved += 1

    print(f"[{time.strftime('%H:%M:%S')}] 保存完了。合計 {total_saved} 件のデータがJSONLとして書き出されました。")

def main():
    # 1. p31_labelsを含むデータをロード
    p31_data = load_data(P31_INPUT_FILE, 'p31_labels')
    
    # 2. aliases情報を含むデータをロード
    aliases_data = load_data(ALIASES_INPUT_FILE, 'aliases')
    
    # 3. データのマージ
    merged_data = merge_data(p31_data, aliases_data)
    
    # 4. JSONL形式でファイルに保存
    save_data(merged_data, OUTPUT_FILE)

if __name__ == "__main__":
    main()