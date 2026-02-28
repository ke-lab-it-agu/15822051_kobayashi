import json

input_file = "wikidata_cl_sorted.jsonl"
output_file = "exist_qid_label_list.jsonl"

count = 0  # 出力したQIDの件数

with open(input_file, "r", encoding="utf-8") as fin,\
    open(output_file, "w", encoding="utf-8") as fout:


    for line in fin:
        try: 
            data = json.loads(line.strip()) 
            for qid, props in data.items(): 
                label = props.get("label") 
                # {"Q31": "Belgium"} の形式で出力 
                json.dump({qid: label}, fout, ensure_ascii=False)
                fout.write("\n") 
                
                count += 1 
                if count % 1_000_000 == 0:
                    print(f"{count} QIDs written...") 
        except json.JSONDecodeError: 
            continue


print(f"Finished. Total QIDs written: {count}")
