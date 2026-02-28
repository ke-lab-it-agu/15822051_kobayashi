import json
import time
from typing import Set

# --- ファイル設定 ---
P31_FILE = "qid_p31_list_simple.jsonl"
LABEL_INPUT_FILE = "exist_qid_label_list.jsonl"
LABEL_OUTPUT_FILE = "p31_optimized_labels.jsonl" # 最適化されたラベルファイル名

def extract_all_used_qids() -> Set[str]:
    """
    P31ファイルから、P31 QID（目的語）のみを集めてユニークなセットを返す。
    （エンティティQID（主語）は除外する）
    """
    used_qids = set()
    start_time = time.time()
    
    print(f"--- 1. P31ファイルから使用されている P31 QID のみを収集 ---")
    
    try:
        with open(P31_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                # entity_qid = data.get("qid") # 主語のQIDはここでは使用しない
                p31_list = data.get("p31", [])
                
                # ⚠️ 変更点: エンティティQID（主語）の収集を削除
                # if entity_qid:
                #     used_qids.add(entity_qid)
                
                # P31 QID（目的語）のみを収集
                for p31_qid in p31_list:
                    used_qids.add(p31_qid)
                
                if (i + 1) % 1000000 == 0:
                    print(f"  [Progress] {i + 1:,} entities processed.")

        print(f"✅ {len(used_qids):,} 件のユニーク P31 QID を収集完了 ({time.time() - start_time:.2f}秒)")
        return used_qids

    except FileNotFoundError:
        print(f"❌ エラー: {P31_FILE} が見つかりません。処理を中断します。")
        return set()
    except Exception as e:
        print(f"❌ エラー (QID収集): {e}")
        return set()


def filter_labels(used_qids: Set[str]):
    """
    全ラベルファイルから、使用されているQIDのラベルのみを抽出して出力する。
    """
    start_time = time.time()
    extracted_count = 0
    
    print(f"\n--- 2. ラベルファイルのフィルタリング ({LABEL_INPUT_FILE} -> {LABEL_OUTPUT_FILE}) ---")
    
    try:
        with open(LABEL_INPUT_FILE, 'r', encoding='utf-8') as fin, \
             open(LABEL_OUTPUT_FILE, 'w', encoding='utf-8') as fout:
            
            for i, line in enumerate(fin):
                data = json.loads(line.strip())
                qid = None
                
                # ユーザー様の形式 {"Q1111": "label"} を想定
                if isinstance(data, dict) and len(data) == 1:
                    key = list(data.keys())[0]
                    if key.startswith("Q") and key[1:].isdigit():
                        qid = key
                
                # 使用されている QID のラベルのみを書き出す
                if qid and qid in used_qids:
                    json.dump(data, fout, ensure_ascii=False)
                    fout.write("\n")
                    extracted_count += 1
                
                if (i + 1) % 1000000 == 0:
                    print(f"  [Progress] {i + 1:,} labels checked.")

    except FileNotFoundError:
        print(f"❌ エラー: {LABEL_INPUT_FILE} が見つかりません。処理を中断します。")
        return
    except Exception as e:
        print(f"❌ エラー (ラベルフィルタリング): {e}")
        return

    print(f"\n✅ ラベルフィルタリング完了 ({time.time() - start_time:.2f}秒)")
    print(f"  - 抽出されたラベル数: {extracted_count:,} 件")
    print(f"  - 出力ファイル: {LABEL_OUTPUT_FILE}")


def main():
    # 1. P31ファイルから P31 QID のみを抽出
    used_qids_set = extract_all_used_qids()
    
    if not used_qids_set:
        print("\n処理を終了します。")
        return

    # 2. ラベルファイルをフィルタリングして最適化ファイルを作成
    filter_labels(used_qids_set)
    
    print("\n--- 完了 ---")
    print(f"これで、{LABEL_OUTPUT_FILE} を使ってDBをロードすれば、P31目的語のラベルのみが登録され、サイズが最適化されます。")

if __name__ == "__main__":
    main()