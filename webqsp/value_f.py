import json
import re

# ファイルのパス
CORRECT_WIKIDATA_FILE_PATH = "../datasets/webqsp/webqsp_test.json"
WIKIDATA_FILE_PATH = "../my_result/webqsp/webqsp_choice_llama.json"

# 正しいWIkidata IDをデータセットから読み込む関数
def read_correct_wikidata_ids(file_path):
    correct_wikidata_ids = []
    
    with open(file_path, "r", encoding="UTF-8") as file:
        data = json.load(file)
        
        for entry in data:
            entities = entry.get("entities", [])
            wikidata_ids = [entity for entity in entities if entity is not None]
            
            if not wikidata_ids:
                correct_wikidata_ids.append([""])
            else:
                correct_wikidata_ids.append(wikidata_ids)
    
    return correct_wikidata_ids

def read_predicted_wikidata_ids(file_path):
    predicted_wikidata_ids = []
    
    with open(file_path, "r", encoding="UTF-8") as file:
        data = json.load(file)
        
        for entry in data:
            entities = entry.get("wikidata_ids", [])
            wikidata_ids = [entity for entity in entities if entity is not None]

            if not wikidata_ids:
                predicted_wikidata_ids.append([""])
            else:
                predicted_wikidata_ids.append(wikidata_ids)

    return predicted_wikidata_ids

def calculate_metrics(correct_wikidata_ids, predicted_wikidata_ids):
    correct_lines = 0
    precision_values, recall_values, f1_score_values = [], [], []

    for i, true_wikidata_ids in enumerate(correct_wikidata_ids, 1):
        if i <= len(predicted_wikidata_ids):
            predicted_wikidata_ids_for_line = predicted_wikidata_ids[i-1]
            
            # print(f"Line {i}: True Wikidata IDs = {true_wikidata_ids}, Predicted Wikidata IDs = {predicted_wikidata_ids_for_line}")
            
            # TP, FP計
            true_positives = len(set(predicted_wikidata_ids_for_line) & set(true_wikidata_ids))
            false_positives = len(set(predicted_wikidata_ids_for_line) - set(true_wikidata_ids))

            if not true_wikidata_ids and not predicted_wikidata_ids:
                true_positives = 1
                false_positives = 0

            # 正しいWikidata IDすべて出力出来たら精度上がる
            if true_positives == len(true_wikidata_ids):
                correct_lines += 1

            # 適合率、再現率、F値計算
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / len(true_wikidata_ids) if len(true_wikidata_ids) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # print(f"Precision = {precision}, Recall = {recall}, F1 Score = {f1_score}")

            precision_values.append(precision)
            recall_values.append(recall)
            f1_score_values.append(f1_score)

    # 精度の分母
    total_lines = min(len(predicted_wikidata_ids), i)
    # 各行の平均適合率、再現率、F値計算
    average_precision = sum(precision_values) / total_lines if total_lines > 0 else 0
    average_recall = sum(recall_values) / total_lines if total_lines > 0 else 0
    average_f1_score = 2 * (average_precision * average_recall) / (average_precision + average_recall) if (average_precision + average_recall) > 0 else 0

    # 精度計算
    accuracy = correct_lines / total_lines if total_lines > 0 else 0

    return accuracy, average_precision, average_recall, average_f1_score

correct_wikidata_ids = read_correct_wikidata_ids(CORRECT_WIKIDATA_FILE_PATH)
predicted_wikidata_ids = read_predicted_wikidata_ids(WIKIDATA_FILE_PATH)
accuracy, precision, recall, f1_score = calculate_metrics(correct_wikidata_ids, predicted_wikidata_ids)

print("\n結果:")
print("精度: {:.3f}".format(accuracy))
print("適合率: {:.3f}".format(precision))
print("再現率: {:.3f}".format(recall))
print("F値: {:.3f}".format(f1_score))