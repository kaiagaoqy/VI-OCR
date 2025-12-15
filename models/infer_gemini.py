
from dotenv import load_dotenv
from retry import retry

import os
from PIL import Image
import argparse
import torch
import json
import requests
from PIL import Image
from io import BytesIO
import re
import os
import tqdm
from PIL import Image
from google import genai

load_dotenv(".env")


client = genai.Client(api_key=os.environ.get("Gemini-API-KEY"))


@retry((Exception), tries=3, delay=0, backoff=0)
def call_gemini(messages, model_name="gemini-1.5-flash", parse_fn=None):
    response = client.models.generate_content(
        model=model_name,
        contents=messages,
        # temperature=0.7,
        # max_output_tokens=512,
        # top_p=0.95,
        # top_k=40,
        # stop_sequences=["\n"],
    )
    ret = ""
    try:
        if parse_fn is not None:
            ret = parse_fn(response.text)
        else:
            ret = response.text
    except:
        print(response.prompt_feedback)
    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str,default='data/mnread/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/gemini_15_flash.json')
    parser.add_argument('--img_dir', type=str,default='data/mnread',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="gemini-1.5-flash")
    parser.add_argument('--use_placeholder', action='store_true',help="Need to self-define placeholder in the question")
    parser.add_argument('--filter', nargs='+',default=["0"], help="low vision filter id")
    parser.add_argument('--acuity_file', type=str,default='data/human/SelectedFilter.csv')
    parser.add_argument('--text_desc', action='store_true',help="Use text description to describe the image")
    
    args = parser.parse_args()
    should_exit = False
    outfile_jsonl = os.path.join(args.outfile,f'{args.model_path.split("/")[-1]}{"_text_desc" if args.text_desc else ""}.jsonl')
    output_file = os.path.join(args.outfile,f'{args.model_path.split("/")[-1]}{"_text_desc" if args.text_desc else ""}.json')
    os.makedirs(args.outfile, exist_ok=True)


    
    samples = json.load(open(args.infile, "r"))['images']
    formatted_samples = []
    q = os.getenv("Prompt")
    if 'etdrs' in args.infile:
        q = q.replace('word','letter')
    args.filter = ["0"] if args.infile != 'data/viocr/anno.json' else args.filter
    # 建议：用 .jsonl 流式写，一条/行
    # outfile_jsonl = args.outfile if args.outfile.endswith(".jsonl") else args.outfile + ".jsonl"

    # 如果需要断点续跑，先读已有的 image_id 集合避免重复（可选）
    done_ids = set()
    if os.path.exists(outfile_jsonl):
        with open(outfile_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_ids.add((obj["image_id"], obj.get("filter", 0)))
                except Exception:
                    pass

    if args.text_desc:
        import pandas as pd
        acuity_df = pd.read_csv(args.acuity_file)
        acuity_df = acuity_df[['Filter_no', 'VA', 'CS']]
        acuity_dict = acuity_df.set_index('Filter_no').to_dict(orient='index')
    with open(outfile_jsonl, "a", encoding="utf-8") as fout:
        for idx, sample in tqdm.tqdm(enumerate(samples), total=len(samples)):
            for filter_id in args.filter:
                key = (int(sample["id"]), int(filter_id) if int(filter_id) > 0 else int(sample.get("Filter_no", 0)))
                if key in done_ids:
                    continue  # 已经算过的跳过，支持断点续跑

                formatted_sample = {"image_id":int(sample["id"]),
                                    "category_id": 1,
                                    "polys":[],
                                    "rec_texts":"",
                                    "rec_score":0,
                                    "det_score":0,
                                    "filter":int(filter_id) if int(filter_id) > 0 else int(sample["Filter_no"]),
                                    }
        
                image_file = os.path.join(args.img_dir,filter_id, sample["file_name"]) if int(filter_id) > 0 else os.path.join(args.img_dir, sample["file_name"])
                if args.text_desc:
                    text_desc = f"You are a low vision reader. Your visual acuity is {acuity_dict[int(formatted_sample['filter'])]['VA']} logMAR and your contrast sensitivity is {acuity_dict[int(formatted_sample['filter'])]['CS']} logCS."
                    q = text_desc + '\n' + q
                    image_file = os.path.join(args.img_dir,'16', sample["file_name"].split("/")[-1])
                if not os.path.exists(image_file):
                    print(f"Image not found: {image_file}")
                    continue
                image = Image.open(image_file)
                messages = [image, q]
                output = ""
                try:
                    output = call_gemini(messages, model_name=args.model_path, parse_fn=lambda x: x.strip().replace(".", ''))
                    formatted_sample["rec_texts"] = output
                except Exception as e:
                    print(f"Error: {e}")
                    should_exit = True
                formatted_sample["rec_texts"] = output
                # === 立刻写入磁盘，避免累计到内存 ===
                fout.write(json.dumps(formatted_sample, ensure_ascii=False) + "\n")
                fout.flush()

                # === 释放临时对象，必要时清理显存 ===
                del messages, output, formatted_sample
   
            
        
    # # os.makedirs(os.path.dirname(args.outfile), exist_ok=True) 
    # json.dump(formatted_samples, open(args.outfile, "w"), indent=4) 
    # # Optionally, if processing completes successfully, remove the resume file.
    # if not should_exit and os.path.exists(resume_file):
    #     os.remove(resume_file)
            
    # 把 .jsonl 转成 .json 数组
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

    jsonl_to_json(outfile_jsonl, output_file)

            

