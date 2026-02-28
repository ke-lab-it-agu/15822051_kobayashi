import json
import ollama
import numpy as np
from tqdm import tqdm
import os
import time
from typing import List, Dict, Tuple, Optional
import sys

# --- 設定 ---
# 入力/出力ファイルのパスとモデルのパラメータを設定します
INPUT_FILE = "final_wikidata_all.jsonl" # ★ 変更: P31ラベル付きの入力ファイル ★
OUTPUT_EMBED = "wikidata_embeddings_06b_ali_classed.dat" # ★ 変更: ファイル名更新 ★
OUTPUT_QID = "wikidata_qids_06b_ali_classed.jsonl" # ★ 変更: ファイル名更新 ★
OUTPUT_LABELS = "wikidata_labels_06b_ali_classed.jsonl" # ★ 変更: ファイル名更新 ★
BATCH_SIZE = 1024
EMBED_DIM = 1024
# 合計行数の推定値 (memmapの初期サイズ決定に使用)
TOTAL_LINES_EST = 100_287_000
MODEL_NAME = "qwen3-embedding:0.6b"
MAX_RETRIES = 5 # Ollama API呼び出しの最大リトライ回数

# ----------------- グローバル変数の初期化 -----------------
qids: List[str] = [] # 追記されたQIDのリスト
labels_dict: Dict[str, str] = {} # QIDとラベルの辞書 (処理済み判定用)
write_index = 0 # 処理再開インデックス

# ----------------- 既存データを読み込み (同期チェック強化) -----------------
def load_existing_jsonl(qid_file: str, label_file: str) -> Tuple[List[str], Dict[str, str], int]:
    """
    JSONLファイルからQIDリストとラベル辞書を再構築し、データの整合性をチェックする。
    memmap、QID、ラベルの3要素が揃っている件数のみをwrite_indexとする。
    ラベルがNoneであっても、QIDファイルに存在すれば処理済みとみなすよう修正。
    """
    loaded_qids = []
    loaded_labels_dict = {}
    
    print(f"[{time.strftime('%H:%M:%S')}] 既存のQID/ラベルファイルをロード中...")

    # 1. QIDファイルの読み込み (各行が {"id": "Qxxx"} JSONLオブジェクト)
    if os.path.exists(qid_file):
        try:
            with open(qid_file, "r", encoding="utf-8") as fq:
                for i, line in enumerate(fq):
                    try:
                        # QIDファイルは {"id": "Qxxx"} 形式のJSONLを想定
                        data = json.loads(line.strip())
                        qid = data.get("id")
                        if qid and isinstance(qid, str) and qid.startswith('Q'):
                            loaded_qids.append(qid)
                        else:
                            # 無効なQIDエントリは警告を出しつつスキップ
                            print(f"⚠️ Warning: Skipped invalid QID entry (Line {i+1}): Missing or malformed 'id'.", file=sys.stderr)
                    except json.JSONDecodeError:
                        # JSONデコードエラーは警告を出しつつスキップ
                        print(f"❌ Error: Skipped invalid JSON line in QID file (Line {i+1}). Content: {line.strip()}", file=sys.stderr)
        except Exception as e:
            print(f"[エラー] {qid_file} の読み込みに失敗しました。エラー: {e}")
            return [], {}, 0

    # 2. ラベルファイルの読み込み (各行が {"Qxxx": "Label"} JSONオブジェクト)
    if os.path.exists(label_file):
        try:
            with open(label_file, "r", encoding="utf-8") as fl:
                for i, line in enumerate(fl):
                    try:
                        data = json.loads(line.strip())
                        # ラベルファイルは {"Qxxx": "Label"} 形式を想定
                        if isinstance(data, dict) and len(data) == 1:
                             loaded_labels_dict.update(data)
                        else:
                            print(f"⚠️ Warning: Skipped invalid Label entry (Line {i+1}): Not a single key-value object.", file=sys.stderr)
                    except json.JSONDecodeError:
                        pass # 処理継続のためJSONデコードエラーは無視
        except Exception as e:
            print(f"[エラー] {label_file} の読み込みに失敗しました。エラー: {e}")
            return [], {}, 0

    # 3. 整合性チェックとインデックス設定 
    final_qids = []
    final_labels_dict = {}
    
    # QIDリストを基準に、ラベルの存在（キーの存在）をチェックし、同期が取れている部分のみを抽出
    for qid in loaded_qids:
        # **修正点**: QIDがラベル辞書にキーとして存在すれば、値がNone（null）でも処理済みとみなす。
        if qid in loaded_labels_dict: 
            final_qids.append(qid)
            # ラベル値（Noneまたは文字列）を保持
            final_labels_dict[qid] = loaded_labels_dict[qid]
        else:
            # QIDファイルにはあるが、ラベルファイルに全くキーが存在しない場合のみ、未完了とみなす
            print(f"[警告] QIDファイルに存在するがラベルファイルにキーがないQIDを検出しました: {qid}。このQID以降は未完了とみなし、同期ポイントで再開します。")
            break

    # 最終的な進行インデックスは、整合性が取れているQIDの数
    write_index = len(final_qids)

    print(f"[{time.strftime('%H:%M:%S')}] 整合性チェック完了: 信頼できる再開インデックスは {write_index} 件です。")
        
    # memmapファイルの整合性チェック
    memmap_size_expected = write_index * EMBED_DIM * 4 # float32 = 4 bytes
    memmap_size_actual = os.path.getsize(OUTPUT_EMBED) if os.path.exists(OUTPUT_EMBED) else 0

    if memmap_size_actual != memmap_size_expected and memmap_size_actual != 0:
        print(f"\n[致命的警告] memmapファイルサイズが不一致 (期待値: {memmap_size_expected} bytes, 実際: {memmap_size_actual} bytes)。再開インデックス {write_index} 件に基づいてエンベディングファイルが上書きされます。")
    
    return final_qids, final_labels_dict, write_index

# 既存データをロードし、再開インデックスを設定
# グローバル変数を更新
qids, labels_dict, write_index = load_existing_jsonl(OUTPUT_QID, OUTPUT_LABELS)
print(f"[{time.strftime('%H:%M:%S')}] 既存データ読み込み完了: {write_index} 件から処理を再開します。")

# ----------------- memmap -----------------
try:
    if os.path.exists(OUTPUT_EMBED):
        # 既存ファイルを開く
        embeddings_memmap = np.memmap(OUTPUT_EMBED, dtype=np.float32, mode="r+", shape=(TOTAL_LINES_EST, EMBED_DIM))
    else:
        # 新規作成
        embeddings_memmap = np.memmap(OUTPUT_EMBED, dtype=np.float32, mode="w+", shape=(TOTAL_LINES_EST, EMBED_DIM))
except Exception as e:
    print(f"[致命的エラー] memmapファイルの初期化に失敗しました: {e}")
    sys.exit(1)


# ----------------- 堅牢な embed -----------------
def embed_batch(texts: List[str]) -> Optional[np.ndarray]:
    """
    ollamaのエンベディングAPIを同期的に呼び出し、指数バックオフでリトライする
    """
    for attempt in range(MAX_RETRIES):
        try:
            # ollama.embed は辞書を返す
            results = ollama.embed(model=MODEL_NAME, input=texts)
            vectors = np.array(results["embeddings"], dtype=np.float32)
            
            if vectors.shape[1] != EMBED_DIM:
                print(f"\n[警告] ベクトル次元が不一致: 期待値 {EMBED_DIM}, 実際 {vectors.shape[1]}。バッチスキップ。", file=sys.stderr)
                return None
            
            return vectors
        
        # **修正点**: ollama.OllamaAPIErrorの代わりに、より一般的なExceptionを捕捉
        except Exception as e:
            print(f"\n[Ollama APIエラー/一般エラー] 試行 {attempt + 1}/{MAX_RETRIES}: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES - 1:
                # 指数バックオフ
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] リトライ上限に達しました。このバッチはスキップされます。", file=sys.stderr)
                return None
    
    return None

# ----------------- バッチ処理 (同期の強制) -----------------
def process_batch(batch_texts: List[str], batch_qids: List[str], batch_labels: List[str], state: Dict, pbar: tqdm):
    """
    単一のバッチを処理し、memmapとJSONLファイルに書き込む。
    成功したベクトル数に基づいて、全ファイルの書き込みを同期させる。
    書き込み後にファイルオブジェクトをフラッシュし、中断に備える。
    """
    
    # 1. エンベディングを生成
    vectors = embed_batch(batch_texts)

    if vectors is None or len(vectors) == 0:
        return

    n_successful = len(vectors)
    
    # 成功したベクトル数に合わせて、QIDとラベルのリストをスライスする
    successful_qids = batch_qids[:n_successful]
    successful_labels = batch_labels[:n_successful]

    if n_successful < len(batch_texts):
        print(f"\n[警告] 部分的な成功: リクエスト数 {len(batch_texts)}, 成功数 {n_successful}。このバッチの残りのデータはスキップされます。", file=sys.stderr)

    
    # 2. memmapへの書き込みとフラッシュ
    start_idx = state['write_index']
    end_idx = start_idx + n_successful

    if end_idx > TOTAL_LINES_EST:
        print(f"\n[致命的エラー] memmapのサイズ ({TOTAL_LINES_EST}) を超える書き込みを試みています ({end_idx})。", file=sys.stderr)
        return

    embeddings_memmap[start_idx:end_idx, :] = vectors
    # memmapに書き込んだ後、すぐにディスクにフラッシュ
    embeddings_memmap.flush()

    # 3. QIDとラベルを厳密に同期を取りながら JSONL で追記書き込み
    
    # QIDファイル (各行に {"id": "Qxxx"} JSON Lines 形式で出力)
    with open(OUTPUT_QID, "a", encoding="utf-8") as fq:
        for qid in successful_qids:
            # {"id": "Qxxx"} 形式で出力
            data = {"id": qid}
            json_line = json.dumps(data, ensure_ascii=False)
            fq.write(json_line + "\n")
            
            qids.append(qid) # メモリ上のQIDリストも更新
        fq.flush() # OSバッファへの書き込みを強制

    # ラベルファイル (各行に {"Qxxx": "Label"} JSONオブジェクト)
    with open(OUTPUT_LABELS, "a", encoding="utf-8") as fl:
        for q, l in zip(successful_qids, successful_labels):
            # {"Qxxx": "Label"} 形式で出力
            # ラベル 'l' が空文字列の場合、JSONでは null にする必要がある。
            # Pythonのjson.dumpsは、Noneをnullに変換する。
            label_value = l if l else None 
            json_line = json.dumps({q: label_value}, ensure_ascii=False)
            fl.write(json_line + "\n")
            labels_dict[q] = label_value # メモリ上のラベル辞書も更新
        fl.flush() # OSバッファへの書き込みを強制

    # 4. 状態更新と進捗表示
    state['write_index'] += n_successful 
    state['processed_batches'] += 1

    pbar.update(n_successful)
    
    elapsed = time.time() - state['start_time']
    current_items = state['write_index'] - write_index # 再開後の処理アイテム数
    
    if current_items > 0:
        avg_time_per_item = elapsed / current_items
        remaining_items = TOTAL_LINES_EST - state['write_index']
        
        # 残り推定時間が巨大な場合は表示を抑制
        if remaining_items > 0:
            eta_sec = avg_time_per_item * remaining_items
            if eta_sec > 3600 * 24 * 30: # 30日以上の場合は表示を簡略化
                 pbar.set_postfix_str(f"ETA: >1 month | Success: {n_successful}/{len(batch_texts)}")
            else:
                 pbar.set_postfix_str(f"ETA: {eta_sec/3600:.2f} hours | Success: {n_successful}/{len(batch_texts)}")
        else:
            pbar.set_postfix_str(f"Completed | Success: {n_successful}/{len(batch_texts)}")
    else:
        pbar.set_postfix_str("Calculating ETA...")

# ----------------- メイン (同期処理) -----------------
def main_sync():
    batch_texts, batch_qids, batch_labels = [], [], []
    state = {"write_index": write_index, "processed_batches": 0, "start_time": time.time()}
    processed_lines_in_input = 0 

    pbar = tqdm(total=TOTAL_LINES_EST, desc="Vectorizing", unit="items", initial=write_index)

    # 入力ファイルの先頭から読み込み直す
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            processed_lines_in_input += 1
            
            try:
                # 入力ファイルは {"Qxxx": {...}} 形式の一つのJSONオブジェクト
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"\n[警告] JSONLのパースエラーが発生しました。行 {processed_lines_in_input} をスキップします。", file=sys.stderr)
                continue 

            for qid, info in data.items():
                
                # 処理済み QID のスキップ (labels_dictに存在するかで判定)
                if qid in labels_dict:
                    # 既に処理済みなので、pbarの初期値に含まれているはずだが、念のため進捗バーを更新（過剰更新だが安全策）
                    # pbar.update(1) # これをすると過剰に増加するので、このループ内では更新しない
                    continue
                
                # --- ベクトル化入力テキストの構造化 ---
                label = info.get("label", "")
                description = info.get("description", "")
                
                p31_labels = info.get("p31_labels", []) 
                aliases = info.get("aliases", []) # aliases情報を取得
                LIMITED_P31_COUNT = 5
                limited_p31_labels = p31_labels[:LIMITED_P31_COUNT]
                LIMITED_ALIASES_COUNT = 3
                limited_aliases = aliases[:LIMITED_ALIASES_COUNT]
                text = ""
                
                # 1. ラベル (*Label*)
                if label:
                    text += f"*{label}* *{label}*"

                # 2. 説明 (:Description)
                if description:
                    # ラベルが存在する場合のみ区切り文字（：）を追加
                    if text:
                        text += "："
                    # 説明は括弧で囲む
                    text += f"({description})"
                
                # 3. P31ラベル (*P31 Label*)
                if limited_p31_labels:
                    p31_str = ", ".join(limited_p31_labels)
                    # 説明またはラベルが存在する場合のみ区切り文字（：）を追加
                    if text:
                        text += "："
                    # P31ラベルはアスタリスクで囲む
                    text += f"*{p31_str}*"
                
                # 4. エイリアス (- [Aliases])
                if limited_aliases:
                    aliases_str = ", ".join(limited_aliases)
                    # 前の要素が存在する場合のみ、長い区切り文字（ - ）を追加
                    if text:
                        text += " - " 
                    # エイリアスは角括弧を使用して囲む
                    text += f"({aliases_str})"
                
                # ----------------------------------------------------
                
                if not text:
                    continue
                    
                batch_texts.append(text)
                batch_qids.append(qid)
                batch_labels.append(label)

                if len(batch_texts) >= BATCH_SIZE:
                    process_batch(batch_texts, batch_qids, batch_labels, state, pbar)
                    # バッチ処理後、リストをリセット
                    batch_texts, batch_qids, batch_labels = [], [], []

    # 残りバッチの処理
    if batch_texts:
        print(f"\n[{time.strftime('%H:%M:%S')}] 最後のバッチ ({len(batch_texts)} 件) を処理中...")
        process_batch(batch_texts, batch_qids, batch_labels, state, pbar)

    pbar.close()
    print(f"\n[{time.strftime('%H:%M:%S')}] 処理完了！合計 {state['write_index']} 件のエンベディングを保存しました。")

# ----------------- 実行 -----------------
if __name__ == "__main__":
    try:
        main_sync()
    except KeyboardInterrupt:
        # 中断時にmemmapが最新の書き込み内容でフラッシュされていることを確認
        embeddings_memmap.flush() 
        print(f"\n[{time.strftime('%H:%M:%S')}] [中断] ユーザーにより処理が中断されました。進行状況は {write_index} 件までディスクに保存されています。")
    except Exception as e:
        print(f"\n[{time.strftime('%H:%M:%S')}] [予期せぬエラー] 処理が停止しました。エラー: {e}")