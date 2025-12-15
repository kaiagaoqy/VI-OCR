import torch
import os
import argparse
import json
import tqdm
from collections.abc import Sequence
import re
from dotenv import load_dotenv
load_dotenv(".env")
torch.manual_seed(1234)


from transformers import AutoModel, AutoTokenizer






def eval_model(tokenizer, model, messages, size):
    # Preparation for inference
    image_file = messages[0]['content'][0]['image']
    prompt = "<image>\nFree OCR. "#messages[0]['content'][1]['text']
    # Tiny: base_size = 512, image_size = 512, crop_mode = False
    # Small: base_size = 640, image_size = 640, crop_mode = False
    # Base: base_size = 1024, image_size = 1024, crop_mode = False
    # Large: base_size = 1280, image_size = 1280, crop_mode = False
    # Gundam: base_size = 1024, image_size = 640, crop_mode = True
    size_dict = {
        "Tiny": (512, 512, False),
        "Small": (640, 640, False),
        "Base": (1024, 1024, False),
        "Large": (1280, 1280, False),
        "Gundam": (1024, 640, True)
    }
    base_size, image_size, crop_mode = size_dict[size]
    output = model.infer(tokenizer, prompt=prompt, image_file=image_file, base_size = base_size, image_size = image_size, crop_mode=crop_mode, save_results = False, test_compress = True, output_path = ' ',eval_mode=True)
    output = output.strip()
    torch.cuda.empty_cache()
    return output




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str,default='data/totaltext/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/totaltext')
    parser.add_argument('--img_dir', type=str,default='data/totaltext',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument('--size', type=str, default="Gundam", choices=["Gundam", "Tiny", "Base", "Small", "Large"])
    parser.add_argument('--filter', nargs='+',default=["1","2","3","4","5","6","7","32","33","34","35","36","38","39","40","41"], help="low vision filter id")

    args = parser.parse_args()
    outfile_jsonl = os.path.join(args.outfile,f'{args.model_path.split("/")[-1]}_{args.size}.jsonl')
    output_file = os.path.join(args.outfile,f'{args.model_path.split("/")[-1]}_{args.size}.json')
    os.makedirs(args.outfile, exist_ok=True)



    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, use_safetensors=True)
    model = model.eval().cuda().to(torch.bfloat16)

    samples = json.load(open(args.infile, "r"))['images']
    output_samples = []
    q = os.getenv("Prompt")
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
                if not os.path.exists(image_file):
                    print(f"Image not found: {image_file}")
                    continue
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_file,
                            },
                            {"type": "text",
                            "text": "<image>\n" + q},
                        ],
                    }
                ]
                output = eval_model(tokenizer, model, messages, args.size)
                formatted_sample['rec_texts'] = output
                # === 立刻写入磁盘，避免累计到内存 ===
                fout.write(json.dumps(formatted_sample, ensure_ascii=False) + "\n")
                fout.flush()

                # === 释放临时对象，必要时清理显存 ===
                del messages, output, formatted_sample
            # output_samples.append(formatted_sample)
    
        
    #     if idx % 100 == 0:
    #         json.dump(output_samples, open(output_file, "w"), indent=4)
    # json.dump(output_samples, open(args.outfile, "w"), indent=4)

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