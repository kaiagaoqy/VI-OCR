### Adjust the format of recognition results to list ###

from typing import Any
import pandas as pd
import json
import re
import os
def strtolist(x):
    """
    Adjust the input string to list
    """
    if isinstance(x, list):
        return x
    try:
        x = x.replace("],\n[", ',')
        x = x.replace("][", ',')
        x = x.replace(", ...", '')
        x = x.replace(", ']", ']')
        x = x.replace('\"]}', "']")
        x = x.replace("'''", "''")
        x = x.replace("\"", "'")
        x = x.replace("```json\n", "")
        x = x.replace("```", "")
        x = x.replace(", and", ",")
        x = x.split(':', 1)[1].strip() if ':' in x else x
        if not re.findall(r'[a-zA-Z0-9]',x):
            return []
        if len(re.findall(r"'.{0,2}'",x)) >20:
            return []
        # if 'sorry', 'cannot', 'assist', 'with', 'that', 'no text found', return []
        if any(word in x.lower() for word in ['sorry', 'cannot', ' no ', 'apologize', 'no text found', 'unanswerable', 'unclear', 'not clear', 'not sure', 'not sure what', 'not sure what you', 'not sure what you\'re', 'not sure what you\'re asking for', 'not sure what you\'re asking for.', 'blurry']):
            return []
        if x.startswith('[') and x.endswith(']'):
            import csv
            import io

            # 去掉方括号
            s_clean = x.strip("[]")

            # 用 csv.reader 解析，quotechar="'" 表示外层引号
            reader = csv.reader(io.StringIO(s_clean), skipinitialspace=True, quotechar="'")
            x = [item for row in reader for item in row]
            # print(tokens)
            # x = eval(x)
        else:
            matcher = re.findall(r"\['.*'\]",x)
            if matcher:
                x = ''
                for group in matcher:
                    x += group
                x = x.replace('][',',')
                x = eval(x)
            else:
                matcher = re.findall(r"'[a-zA-Z0-9 ]{1,15}'",x)
                if matcher:
                    x = []
                    for group in matcher:
                        x += eval(group).split()
                else:
                    if ',' in x and len(x.split(',')[0]) < 15:
                        x = [i.strip(',') for i in x.split() if i]
                    else:
                        if len(x) < 15:
                            x = x.replace(' ','').replace('unanswerable','').split()
                            if isinstance(x, str):
                                x = [i for i in x if i]                           
                        else:
                            x = [] if len(x.split()) > 12 else [i.strip() for i in x.split() if i.strip()]
    
                
    except Exception as e:
        print(f"Error processing string {x}: {e}")
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            matcher = re.findall(r"([a-zA-Z'])'([a-zA-Z])",x)
            if matcher:
                for group in matcher:
                    x = x.replace("'".join(group), "\\'".join(group))
            return eval(x)
        else:
            return x
    #only keep alphanumeric and digits characters
    x = [re.sub(r'[^a-zA-Z0-9]', '', i) for i in x if i]
    if len(x) == 1 and 'etdrs' in FOLDER:
        x = list(x[0].strip())
    return x

def jsonl_to_json(jsonl_path, json_path):
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(json_path, "w", encoding="utf-8") as fout:
        fout.write("[\n")
        first = True
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if not first:
                fout.write(",\n")
            fout.write(line)
            first = False
        fout.write("\n]\n")

if __name__ == "__main__":
    import glob
    import json
    import pandas as pd
    import tqdm

    FOLDER = 'totaltext'
    files = glob.glob(f'outputs/*/all_results.json')
    for file in tqdm.tqdm(files, total=len(files)):
        try:
            model_name = file.split('/')[1]
            out_path = f'outputs/filtered_results/{model_name}.json'
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(file) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            if 'deepseek' in file.lower():
                df['rec_texts'] = df['rec_texts'].apply(lambda x: re.sub(r'[ \n\\]', ', ', x) if isinstance(x, str) else x)
            df['rec_texts'] = df['rec_texts'].apply(lambda x: strtolist(x))
            # df['rec_texts'] = df['rec_texts'].apply(lambda x: [''.join(x)])
            df = df.to_dict(orient='records')
            with open(out_path, 'w') as f:
                json.dump(df, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue