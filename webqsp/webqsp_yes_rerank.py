import os
import json
import numpy as np
import faiss
import ollama
import sqlite3
import unicodedata
import re
from typing import List, Any
from pydantic import BaseModel, field_validator
from langchain_core.prompts import ChatPromptTemplate
import time
import random

VECTOR_MODEL="qwen3-embedding:0.6b"
LLM_MODEL="qwen2.5:7b"

# ============================================
# 0. DB設定
# ============================================

DB_NAME = "../wikidata/wikidata_p31_map.db"


def fix_unicode_escapes(text: str) -> str:
    try:
        return text.encode("utf-8").decode("unicode_escape")
    except Exception:
        return text
def normalize_label(text: str) -> str:
    if not text:
        return ""
    text = fix_unicode_escapes(text)
    
    # Unicode 正規化（アクセント分離）
    text = unicodedata.normalize("NFKD", text)
    # アクセント除去
    text = "".join(c for c in text if not unicodedata.combining(c))
    # 小文字化
    text = text.lower()
    # 記号・余分な空白の除去
    # 記号の軽い正規化（削除しない）
    text = text.replace("×", "x")
    text = text.replace("–", "-").replace("—", "-")
    
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================
# 1. データ読み込み
# ============================================

BASE_DIR = "../wikidata"
EMBED_PATH = os.path.join(BASE_DIR, "wikidata_embeddings_06b_ali_classed.dat")
EMBED_DIM = 1024

# QID読み込み (JSONL形式に対応)
QID_NPY_PATH = os.path.join(BASE_DIR, "wikidata_qids.npy")

print("Loading qids (mmap):", QID_NPY_PATH)
qids = np.load(QID_NPY_PATH, mmap_mode="r")
n_vectors = len(qids)

print(f"Successfully loaded {n_vectors} QIDs.")


n_vectors = len(qids) # ベクトルの総数
print(f"Successfully loaded {n_vectors} QIDs.")

print("Loading embeddings:", EMBED_PATH)
# memmapを使用して埋め込みデータをメモリにマッピング

if os.path.exists(EMBED_PATH):
    # QIDの総数に基づき、埋め込みデータをメモリにマッピング
    print(f"    -> Using {n_vectors} vectors from the start of the embeddings file.")
    embeds = np.memmap(EMBED_PATH, dtype=np.float32, mode="r", shape=(n_vectors, EMBED_DIM))
else:
    raise FileNotFoundError(f"Embeddings file not found at {EMBED_PATH}. Please run the vectorization script first.")


# ラベル読み込み (JSONL形式: 1行に1つの {"Qxxx": "Label"} JSONオブジェクト)
LABEL_NPY_PATH = os.path.join(BASE_DIR, "wikidata_labels.npy")

print("Loading labels:", LABEL_NPY_PATH)
labels = np.load(LABEL_NPY_PATH, allow_pickle=True).item()
        

# FAISS index 構築（IndexIVFFlatに変更）
# ハイパーパラメータ
N_LIST = 1000  # クラスタ数
N_PROBE = 100   # 検索時にチェックするクラスタ数 (速度と精度のトレードオフ)
M_PQ = 128          # PQ 分割数（最重要）
NBITS = 8          # 各 subvector 8bit

# インデックスファイルのパスをIVF用に変更
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_ali_class_06b_ivf.index") 

if os.path.exists(FAISS_INDEX_PATH):
    print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
    # 既存のインデックスをロード
    index = faiss.read_index(FAISS_INDEX_PATH,faiss.IO_FLAG_MMAP)
    # 検索設定: ロード後も nprobe を設定
    index.nprobe = N_PROBE 
    print("✅ FAISS index loaded successfully.")
else:
    print("FAISS index file not found. Building IVF index...")
    
    # 1. データをクラスタリングするための基盤（量子の設定）
    quantizer = faiss.IndexFlatL2(EMBED_DIM)
    
    # 2. IVFインデックスを作成: データの分割と格納
    index = faiss.IndexIVFPQ(
        quantizer,
        EMBED_DIM,
        N_LIST,
        M_PQ,
        NBITS
    )
    # 3. 訓練 (Training): データのサブセットを使ってクラスタ中心を決定
    print("Starting FAISS Index Training (IVF)...")
    # 訓練には全データは不要。ここでは最初の100万件を使用
    train_data = embeds[:min(1000000, n_vectors)]
    index.train(train_data) 
    print("Training complete.")

    # 4. データをインデックスに追加（分割 add）
    BATCH = 100_000  # 10万ずつ（メモリ厳しければ 50_000）

    print("Starting FAISS add (batched)...")
    start_time = time.time()

    for i in range(0, n_vectors, BATCH):
        end = min(i + BATCH, n_vectors)
    
        print(
            f"Adding vectors {i:,} - {end:,} / {n_vectors:,} "
            f"({end / n_vectors:.2%})"
        )
    
        batch = np.ascontiguousarray(embeds[i:end])
        index.add(batch)

    elapsed = time.time() - start_time
    print(f"✅ Finished adding vectors. Total time: {elapsed/60:.1f} min")
    
    # 5. 検索設定: 検索時にチェックするクラスタ数を設定
    index.nprobe = N_PROBE 
    print("✅ FAISS index built successfully.")
    
    # 6. 構築後、インデックスを保存
    print(f"Saving FAISS index to: {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("✅ FAISS index saved.")

# --- 整合性チェック（追加） ---
print("\n--- 最終整合性チェック ---")
if index.ntotal == n_vectors:
    print(f"✅ OK: FAISS登録件数 ({index.ntotal}) とQIDリスト件数 ({n_vectors}) が一致しています。")
else:
    raise ValueError(f"❌ 致命的な不整合: FAISS登録件数 ({index.ntotal}) とQIDリスト件数 ({n_vectors}) が一致しません。処理を中断します。")

if len(labels) == n_vectors:
    print(f"✅ OK: ラベル辞書件数 ({len(labels)}) とQIDリスト件数 ({n_vectors}) が一致しています。")
else:
    print(f"⚠️ 警告: ラベル辞書件数 ({len(labels)}) とQIDリスト件数 ({n_vectors}) が不一致です。")
    
# ============================================
# 2. Pydantic モデルで mentions 正規化
# ============================================

class MentionOutput(BaseModel):
    mentions: List[str] = []

    @field_validator("mentions", mode="before")
    def flatten_mentions(cls, v: Any) -> List[str]:
        flat_list = []
        if isinstance(v, list):
            for item in v:
                # ネストされたリストをフラット化
                if isinstance(item, list):
                    flat_list.extend([str(x) for x in item])
                else:
                    flat_list.append(str(item))
        return flat_list

MAX_MENTIONS = 5 # 抽出上限

# ============================================
# 3. Few-shot メンション抽出
# ============================================

mention_examples = [
    ("How long does it take?", {"mentions": []}),
    ("Provide the title of a classical music album", {"mentions": ["classical music"]}),
    ("How many people lived in Bourg-en-Bresse at the beginning of 2015?", {"mentions": ["Bourg-en-Bresse"]}),
    ("what language is spoken in kirikou and the wild beasts", {"mentions": ["kirikou and the wild beasts"]}),
    ("What character did John Noble play in Lord of the Rings?", {"mentions": ["John Noble", "Lord of the Rings"]}),
    ("What capital of Australia?", {"mentions": ["Australia"]})
]

few_shot_messages = []
for q, a in mention_examples:
    few_shot_messages.append(("human", q))
    few_shot_messages.append(("assistant", json.dumps(a["mentions"], ensure_ascii=False)))

mention_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert at extracting named entities from questions.Return only meaningful entity mentions. Do not answer the question in the utterance."),
        *few_shot_messages,
        ("human", "{text}")
    ]
)

def extract_mentions(text: str) -> List[str]:
    def call_llm(extra_system_msg=None):
        messages = mention_prompt_template.messages
        if extra_system_msg:
            messages = [
                ("system", extra_system_msg),
                *messages[1:]
            ]
        prompt = ChatPromptTemplate.from_messages(messages).format(text=text)
        res = ollama.generate(model=LLM_MODEL, prompt=prompt)
        try:
            return json.loads(res["response"])
        except Exception:
            return []

    # 1st call
    raw_mentions = call_llm()

    def filter_mentions(raw):
        text_lower = text.lower()
        out = []
        for m in raw:
            if isinstance(m, str) and m.lower() in text_lower:
                out.append(m)
        return list(dict.fromkeys(out))

    filtered = filter_mentions(raw_mentions)

    # ===============================
    #  条件付き 再呼び出し
    # ===============================
    if raw_mentions and not filtered:
        raw_mentions = call_llm(
            extra_system_msg=(
                "You MUST extract entity mentions that appear EXACTLY "
                "as substrings in the question text. "
                "Do NOT paraphrase, infer, or normalize names."
            )
        )
        filtered = filter_mentions(raw_mentions)

    return filtered[:MAX_MENTIONS]

# ==========================
# SQLite グローバル接続
# ==========================
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("PRAGMA journal_mode = OFF;")
cursor.execute("PRAGMA synchronous = OFF;")
cursor.execute("PRAGMA temp_store = MEMORY;")

def get_p31_labels_bulk(qids: list[str]) -> dict[str, list[str]]:
    """
    複数 QID の P31 ラベルを一括で取得
    return: { "Qxxx": ["city", "administrative territorial entity"] }
    """
    if not qids:
        return {}

    placeholders = ",".join("?" for _ in qids)

    query = f"""
        SELECT
            T1.entity_qid,
            T2.label
        FROM p31_map AS T1
        JOIN labels AS T2 ON T1.p31_qid = T2.qid
        WHERE T1.entity_qid IN ({placeholders});
    """

    cursor.execute(query, qids)

    p31_map = {}
    for entity_qid, label in cursor.fetchall():
        p31_map.setdefault(entity_qid, []).append(label)

    return p31_map

# ============================================
# 4. 補助文章生成
# ============================================
def generate_helper_text(mention: str, utterance: str, predicted_p31: List[str]) -> str:

    # 予測されたP31ラベルをプロンプト用に結合
    p31_str = ", ".join(predicted_p31)

    prompt = f"""
    You are an expert in explaining mentions in a sentence.
    Generate a sentence that acts as a short, concise Wikidata-style description of the Mention.
    Using general encyclopedic knowledge.

    ### Constraint
    DO NOT answer the question asked in the Utterance. 
    limited to a single, concise sentence (max 15 words) in **English**. 
    
    Utterance: "{utterance}"
    Mention: "{mention}"
    Predicted Semantic Types: {p31_str}
    
    Return ONLY the helper_text.
    """
    
    # Ollama API呼び出し
    res = ollama.generate(model="gemma2:9b", prompt=prompt)
    return res["response"].strip()
# ============================================
# 5. メンションの分類ラベル予測 
# ============================================

def predict_p31_labels(mention: str, utterance: str) -> List[str]:
    
    prompt = f"""
You are an expert Wikidata semantic type classifier (P31: instance of).

Your task is to infer the MOST CONTEXTUALLY APPROPRIATE P31 label for the given mention,
based strictly on the meaning of the utterance.

### STRICT OUTPUT RULES
1. Output **ONE** label.
2. Output must be **English** only.
3. Output must be a **valid Wikidata P31 label** such as:
   - human
   - city
   - nation
   - organization
   - film
   - album
   - company
   - language
   - character in ~
   - series
   - capital
   and so on.
   
4. If the mention is ambiguous with no strong signals, output:
   - entity

fewshot
[
  {{
    "utterance": "What is the local dialect of Occitania",
    "mention": "Occitania",
    "p31_label": "region"
  }},
  {{
    "utterance": "Which company was established by Steve Jobs?",
    "mention": "Steve Jobs",
    "p31_label": "human"
  }},
  {{
    "utterance": "what language is Raamdhenu in",
    "mention": "Raamdhenu",
    "p31_label": "film"
  }},
]

### Context
Utterance: "{utterance}"
Mention: "{mention}"

### Final Output
Return ONLY the predicted P31 label:
"""
    # Ollama API呼び出し
    res = ollama.generate(model="gemma2:9b", prompt=prompt)
    
    # レスポンスをパースしてリストにする
    raw_response = res["response"].strip()
    
    # コンマで分割し、各要素から不要な空白を削除
    predicted_labels = [label.strip() for label in raw_response.split(',') if label.strip()]
    
    # 予期せぬエラーでリストが空になった場合、安全策として "entity" を返す
    if not predicted_labels:
        return ["entity"]
        
    return predicted_labels[:3] # 最大3つに制限

# ============================================
# 5. FAISS検索
# ============================================

def retrieve_candidates(mention: str, utterance: str, predicted_p31: List[str], helper_text: str, topk: int = 20):
    
    # 予測されたP31ラベルを検索クエリ用にコンマ区切りの文字列に結合
    predicted_p31_str = ", ".join(predicted_p31)

    # 2. 検索クエリを構造化
    # データポイントの構造 (Label (Description) — Aliases: ...) とアラインさせる
    structured_query_text = (
        f"**{mention}**:({helper_text}):*{predicted_p31_str}* — {utterance}"
    )

    # 3. 単一の構造化されたプロンプトとしてベクトル化
    # エンベディング API呼び出しに指数バックオフを適用
    for i in range(3): # 3回までリトライ
        try:
            vec_res = ollama.embed(model=VECTOR_MODEL, input=[structured_query_text])
            vec = np.array(vec_res["embeddings"][0]).astype(np.float32).reshape(1, -1)
            break
        except Exception as e:
            if i < 2:
                time.sleep(2 ** i)
                print(f"[WARN] Embedding retry {i+1} for {mention}. Error: {e}")
            else:
                # 最終的に失敗した場合、ゼロベクトルで続行（検索結果は悪化する）
                print(f"[ERROR] Failed to embed query after multiple retries: {e}")
                vec = np.zeros((1, EMBED_DIM), dtype=np.float32)

    D, I = index.search(vec, topk)

    results = []
    candidate_qids = []

    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        qid = qids[idx]
        candidate_qids.append(qid)
        results.append({
            "qid": qid,
            "distance": float(dist),
            "label": labels.get(qid, "LABEL_NOT_FOUND")
        })

    # ★ ここで一括 P31 取得
    p31_map = get_p31_labels_bulk(candidate_qids)

    for c in results:
        c["p31"] = p31_map.get(c["qid"], [])

    return results

# ============================================
# 6. LLM rerank
# ============================================

def rank_best_entity(mention: str, utterance: str, candidates: List[dict], helper_text: str, predicted_p31: List[str]) -> str:
    # P31ラベルを候補テキストに追加するために結合
    cand_lines = []
    for c in candidates:
        # P31ラベルを取得 (DB呼び出し)
        
        cand_lines.append({
            "qid": c["qid"],
            "label": c["label"],
            "label_norm": normalize_label(c["label"]),
            "p31": ", ".join(c["p31"]),
            "distance": round(c["distance"], 4),
            "surface_match": normalize_label(c["label"]) == normalize_label(mention)
        })

    candidates_json = json.dumps(cand_lines, ensure_ascii=False, indent=2)
    
    # 予測されたP31ラベルをプロンプト用にコンマ区切りの文字列に結合
    predicted_p31_str = ", ".join(predicted_p31)

    prompt = f"""
You are a professional Entity Linking expert.

Link the Mention to the correct Wikidata entity. This is Entity Linking.

Context
Utterance: {utterance}
Mention: {mention}
Helper Description: {helper_text}
Predicted Semantic Type (P31): {predicted_p31_str}

Candidates
{candidates_json}

You MUST always select EXACTLY ONE QID from the candidates.
Consider the context, mention, Semantic Type (P31), distance(lower is better), and helper_text.

If multiple candidates are plausible, rank them using the following priorities.

### Decision Rules (MUST FOLLOW STRICTLY)

1. Surface-form match WITH semantic validity
Exact name(Surface-form) or Alias matches are strong evidence
ONLY IF the entity’s P31 and meaning fit the utterance context.

2.Semantic consistency (P31)
The entity’s P31 must align with how the mention is used in the utterance.

3.Distance
Prefer the smallest distance only after considering name and semantic factors.

note:
label_norm is a normalized form of the entity name with accents removed and lowercased.

### Output Format (STRICT)
Output EXACTLY ONE QID, for example:
Q12345

Do NOT output anything else.
"""

    res = ollama.generate(model="gemma2:9b", prompt=prompt)
    best_qid = res["response"].strip()
    
    # 安全策: QID形式でなければ空文字
    if not best_qid.startswith("Q"):
        best_qid = ""
    return best_qid


# ============================================
# 7. WebQSP 全件処理
# ============================================

def run_el_pipeline(input_path: str, output_path: str):
    print("Loading input:", input_path)
    try:
        with open(input_path, "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    
    # カウンターの初期化
    total_questions = len(dataset) # 総問題数
    questions_hit_count = 0        # 質問レベルでのヒット数 (質問に紐づくQIDのいずれかが候補に含まれた回数)
    total_mentions = 0  # 処理した全メンションの数
    candidate_hit_count = 0  # 正解QIDが候補リスト (topk) に含まれていた回数

    # 出力ファイルの初期化
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("[\n")

    for idx, item in enumerate(dataset, start=1):
        text = item["utterance"]
        
        question_hit_flag = False
        # === ターゲットQIDの読み込みを "entities" キーに変更 ===
        if "entities" in item and isinstance(item["entities"], list):
            # QIDリストを直接取得
            target_qids = [qid.strip('"') for qid in item["entities"] if qid is not None]
        else:
            # "entities" キーが見つからない、またはリストでない場合は空リストとする
            target_qids = set()
        # =======================================================
        print("=" * 60)
        print(f"[{idx}] Utterance: {text}")

        mentions = extract_mentions(text)
        print("    mentions:", mentions)

        wikidata_ids = []

        for m in mentions:
            total_mentions += 1
            # 予測されたP31ラベルを取得 (リストとして返される)
            predicted_p31 = predict_p31_labels(m, text)
            print(f"Predicted P31: {predicted_p31}")
            
            helper_text =  generate_helper_text(m, text,predicted_p31)
            print(f"    Helper text: {helper_text}")
            
            # リストをそのまま渡す
            candidates = retrieve_candidates(m, text, predicted_p31, helper_text)
            candidate_qids = {c['qid'] for c in candidates}
            is_hit=False
            common_qids=set()
            
            # ターゲットQIDと候補QIDの共通部分を計算
            if target_qids:
                common_qids = set(target_qids) & candidate_qids
                
                if common_qids:
                    candidate_hit_count += 1
                    is_hit = True
                    question_hit_flag =True
            
            # デバッグ情報にP31を含めて出力
            print("Candidates:")
            for c in candidates:
                # P31ラベルを取得 
                p31_str = ", ".join(c["p31"])
                label_raw = c.get("label", "")
                label_norm = normalize_label(label_raw)
                print(f"        - {c['qid']}: {label_norm}(P31: {p31_str}) (distance={c['distance']:.4f})")

            print(f" Hit in Top-K Candidates: {'✅ YES' if is_hit else '❌ NO'} (Found QIDs: {', '.join(common_qids) if common_qids else 'None'})")
            # リストをそのまま渡す
            best = rank_best_entity(m, text, candidates, helper_text, predicted_p31)
            wikidata_ids.append(best)
            print(f"    -> {m}     =>    {best}")
            
            
        if question_hit_flag:
            questions_hit_count +=1
        # --- ログ出力の追加: 現在の進捗 (ここを修正) ---
        current_recall = questions_hit_count / idx if idx > 0 else 0
        print(f" 候補ヒット数/処理済問題数: {questions_hit_count}/{idx} ({current_recall:.4f})")

        entry = {
            "index": idx,
            "entities_text": mentions,
            "wikidata_ids": [wid.strip('"') for wid in wikidata_ids]
        }

        with open(output_path, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            if idx < len(dataset):
                f.write(",\n")
            else:
                f.write("\n")

    with open(output_path, "a", encoding="utf-8") as f:
        f.write("]")

    print("\nDONE! →", output_path)

# ============================================
# 8. 実行
# ============================================

INPUT_FILE_PATH = "../datasets/webqsp/webqsp_test.json"
OUTPUT_FILE_PATH = "../my_result/webqsp/webqsp_yes_rerank.json"

if __name__ == "__main__":
    run_el_pipeline(INPUT_FILE_PATH, OUTPUT_FILE_PATH)