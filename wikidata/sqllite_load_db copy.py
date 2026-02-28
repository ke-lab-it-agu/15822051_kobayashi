import sqlite3
import json
import time

# --- ファイル設定 ---
# フィルタリング後のラベルファイル名
LABEL_FILE = "p31_optimized_labels.jsonl"
# P31リストファイル名
P31_FILE = "qid_p31_list_simple.jsonl" 
DB_NAME = "wikidata_p31_map.db"

def load_labels(conn):
    """p31_optimized_labels.jsonl を読み込み、'labels' テーブルを作成する"""
    cursor = conn.cursor()
    print("--- 1. labels テーブルの作成とロード ---")
    
    # テーブル作成: QIDとラベルを格納
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            qid TEXT PRIMARY KEY,
            label TEXT
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_qid ON labels (qid);")
    
    insert_data = []
    total_loaded = 0
    
    try:
        with open(LABEL_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                qid = None
                label = None
                
                if isinstance(data, dict) and len(data) == 1:
                    key = list(data.keys())[0]
                    if key.startswith("Q") and key[1:].isdigit():
                        qid = key
                        label = data[key]
                
                if qid and label:
                    insert_data.append((qid, label))
                    total_loaded += 1
                
                if len(insert_data) >= 100000:
                    cursor.executemany("INSERT OR REPLACE INTO labels (qid, label) VALUES (?, ?)", insert_data)
                    insert_data = []
                    # ログの頻度を上げる
                    # if (i + 1) % 1000000 == 0:
                    #     print(f"  [Progress] {total_loaded:,} labels loaded...")
    
        if insert_data:
            cursor.executemany("INSERT OR REPLACE INTO labels (qid, label) VALUES (?, ?)", insert_data)
        
        conn.commit()
        print(f"✅ 'labels' テーブルに {total_loaded:,} 件のデータをロード完了。")

    except FileNotFoundError:
        print(f"❌ エラー: {LABEL_FILE} が見つかりません。スキップします。")
    except Exception as e:
        print(f"❌ エラー (labels): {e}")


def load_p31_map(conn):
    """qid_p31_list_simple.jsonl を読み込み、'p31_map' テーブルを作成する"""
    cursor = conn.cursor()
    print("\n--- 2. p31_map テーブルの作成とロード ---")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS p31_map (
            entity_qid TEXT NOT NULL,
            p31_qid TEXT NOT NULL
        );
    """)
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_p31_unique ON p31_map (entity_qid, p31_qid);")

    insert_data = []
    total_entity_count = 0 # ログ用
    
    try:
        with open(P31_FILE, 'r', encoding='utf-8') as f:
            # --- ここにデバッグログを追加 ---
            print(f"  [DEBUG] {P31_FILE} の読み込みを開始しました。最初の進捗ログまでしばらくお待ちください...")
            
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                entity_qid = data.get("qid")
                p31_list = data.get("p31", [])
                total_entity_count += 1
                
                if entity_qid and p31_list:
                    # リストを分解し、1対多のペアに変換
                    for p31_qid in p31_list:
                        insert_data.append((entity_qid, p31_qid))
                
                # 10万ペアごとにバッチ挿入
                if len(insert_data) >= 100000:
                    cursor.executemany("INSERT OR IGNORE INTO p31_map (entity_qid, p31_qid) VALUES (?, ?)", insert_data)
                    insert_data = []
                    # ログの頻度を上げて進捗をわかりやすくする
                    # 50万から10万エンティティごとに変更
                    if (i + 1) % 100000 == 0: 
                        print(f"  [Progress] {total_entity_count:,} entity entries processed. Committing batch...")
                        conn.commit() # 進捗確認のために強制的にコミットする
    
        # 残りのデータを挿入
        if insert_data:
            cursor.executemany("INSERT OR IGNORE INTO p31_map (entity_qid, p31_qid) VALUES (?, ?)", insert_data)
        
        conn.commit()
        print(f"✅ 'p31_map' テーブルのロード完了。")

    except FileNotFoundError:
        print(f"❌ エラー: {P31_FILE} が見つかりません。スキップします。")
    except Exception as e:
        print(f"❌ エラー (p31_map): {e}")

def main():
    start_time = time.time()
    
    # データベースに接続（ファイルが存在しない場合は新規作成）
    conn = sqlite3.connect(DB_NAME)
    
    # データをロード
    load_labels(conn)
    load_p31_map(conn)
    
    conn.close()
    
    print(f"\n--- 完了 ---")
    print(f"データベース名: {DB_NAME}")
    print(f"全処理時間: {time.time() - start_time:.2f} 秒")
    print("これで、SQLクエリによるデータ結合が可能です。")

if __name__ == "__main__":
    main()