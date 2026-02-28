import json
import time
from typing import Dict, Any, Optional
import requests # 👈 外部APIアクセスに必要
import re

# ====================================================================
# 1. Wikidata API アクセス設定とヘルパー関数
# ====================================================================

# WikidataのMediaWiki APIエンドポイント
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php" 

# カスタムUser-Agentを設定
# 多くのAPIは、ボットやスクリプトが識別できるようにUser-Agentを要求します。
# これがないと403 Forbiddenになることがあります。
HEADERS = {
    'User-Agent': 'NullLabelFixerBot/1.0 (contact: your-email@example.com)'
}


def is_qid_valid_and_small(qid: str) -> bool:
    """
    QIDが 'Q' で始まり、かつ数値部分が100000以下であるかチェックする。
    """
    if not qid.startswith('Q'):
        return False
    try:
        num = int(qid[1:])
        return num <= 100000
    except ValueError:
        return False

def get_wikidata_label(qid: str) -> Optional[str]:
    """
    Wikidata APIを使用して、指定されたQIDの英語ラベルを取得する。
    レートリミット対策として指数バックオフを実装。
    
    注意: 実際のリクエストには大量のトラフィックが発生する可能性があります。
    """
    params = {
        'action': 'wbgetentities',
        'ids': qid,
        'props': 'labels',
        'languages': 'en', # <-- 英語ラベルのみを取得
        'format': 'json'
    }
    
    retries = 3
    delay = 1  # 初期遅延時間（秒）

    for i in range(retries):
        # ⚠️ APIリクエストを有効化します
        try:
            print(f" [API 呼び出し中...] QID {qid} の英語ラベルを検索 (試行 {i+1})")
            
            # User-Agentヘッダーを含めてリクエストを実行
            response = requests.get(WIKIDATA_API_ENDPOINT, params=params, headers=HEADERS)
            response.raise_for_status() # HTTPエラー（4xx, 5xx）をチェック
            
            data = response.json()
            # 取得したJSONから英語ラベルを抽出
            label = data.get('entities', {}).get(qid, {}).get('labels', {}).get('en', {}).get('value')
            return label
        
        except requests.RequestException as e:
            # ネットワークエラーやHTTPエラーが発生した場合
            print(f" APIリクエストエラー: {e}. {delay}秒後に再試行します...")
            time.sleep(delay)
            delay *= 2  # 指数バックオフ
        except Exception as e:
            # その他の予期せぬエラー
            print(f"予期せぬエラー: {e}")
            break
        
    return None # すべての再試行が失敗した場合
            
# ====================================================================
# 2. JSONL 処理と修正パイプライン
# ====================================================================

def fix_null_labels_in_jsonl(input_file: str, output_file: str):
    """
    JSONLファイルを読み込み、QID <= 100000 かつ labelがnullのエントリを修正して出力する。
    """
    
    print(f"--- 処理開始: {input_file} -> {output_file} ---")
    
    total_entries = 0
    fixed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total_entries += 1
            
            try:
                # 1. 行をJSONとしてパース
                data = json.loads(line)
                
                # 2. 辞書を反復処理し、修正対象を特定
                is_line_modified = False
                
                # dataは {QID: {label: ..., ...}} の形式なので、QIDと値のペアで反復
                for qid, entity_data in data.items():
                    
                    # 3. 修正条件のチェック
                    is_null_label = entity_data.get('label') is None
                    is_small_qid = is_qid_valid_and_small(qid)

                    if is_null_label and is_small_qid:
                        
                        print(f" 🔍 修正対象発見: {qid} - label: null")
                        
                        # 4. Wikidata APIからラベルを取得
                        new_label = get_wikidata_label(qid)
                        
                        if new_label:
                            # 5. ラベルを更新
                            entity_data['label'] = new_label
                            is_line_modified = True
                            fixed_count += 1
                            print(f" ✅ 修正完了: {qid} -> Label: '{new_label}'")
                        else:
                            print(f" ❌ ラベル取得失敗: {qid} の英語ラベルを取得できませんでした。")
                    
                    # 修正が検出された場合、元の辞書を更新 (data[qid] = entity_data は不要、参照渡しで更新される)
                
                # 6. 修正された (またはされていない) エントリを出力ファイルに書き込み
                json_line = json.dumps(data, ensure_ascii=False)
                outfile.write(json_line + '\n')

            except json.JSONDecodeError:
                print(f" [エラー] 行 {total_entries} が有効なJSONではありません。スキップします。")
                outfile.write(line) # 元の行をそのまま出力して破損を防ぐ
                continue
            except Exception as e:
                print(f" [予期せぬエラー] 行 {total_entries}: {e}")
                outfile.write(line)
                continue

            if total_entries % 100 == 0:
                print(f"\n--- 処理中: {total_entries}行完了 | 修正済み: {fixed_count}件 ---")
    
    print("\n" + "="*50)
    print(f"✨ 処理完了サマリー ✨")
    print(f"総エントリ数: {total_entries}件")
    print(f"修正されたエントリ数: {fixed_count}件")
    print(f"出力ファイル: {output_file}")
    print("="*50)

# ====================================================================
# 3. 実行部分
# ====================================================================

# 入力ファイルと出力ファイルのパスを設定してください
INPUT_FILE = "wikidata_alias_p31_m.jsonl"
OUTPUT_FILE = "final_wikidata_all.jsonl"

if __name__ == "__main__":
    # === ⚠️ APIリクエスト有効化後のテスト実行用コード（ファイルが存在する場合） ===
    # 実際の実行時には、この if ブロックが動作します。
    # 実行前に INPUT_FILE (例: wikidata_alias_p31_m.jsonl) が存在するか確認してください。
    
    # 既存のファイルがない場合のテスト用ダミーファイル生成
    import os
    if not os.path.exists(INPUT_FILE):
        print(f"--- ⚠️ WARNING: 入力ファイル '{INPUT_FILE}' が見つかりません。テスト用ダミーファイルを生成します。 ---")
        demo_data = [
            {"Q207": {"label": None, "description": "43rd President", "p31_labels": ["human"]}}, 
            {"Q500000": {"label": None, "description": "Too large QID", "p31_labels": ["concept"]}}, 
            {"Q1": {"label": None, "description": "everything", "p31_labels": ["concept"]}}, 
            {"Q2": {"label": "Earth", "description": "home planet", "p31_labels": ["planet"]}}
        ]
        
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            for item in demo_data:
                f.write(json.dumps(item) + '\n')
        print(f"--- テスト用ファイル '{INPUT_FILE}' を生成しました。 ---")

    fix_null_labels_in_jsonl(INPUT_FILE, OUTPUT_FILE)