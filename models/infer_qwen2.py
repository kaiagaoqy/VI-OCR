import torch
import os
import argparse
import json
import tqdm
from PIL import Image
from collections.abc import Sequence
import re
from dotenv import load_dotenv
load_dotenv(".env")
torch.manual_seed(1234)


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info






def eval_model(processor, model, messages):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
    torch.cuda.empty_cache()
    return output[0]




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str,default='data/viocr/anno.json',help="Json file storing image paths and annotations")
    parser.add_argument('--outfile', type=str,default='output/totaltext_all')
    parser.add_argument('--img_dir', type=str,default='data/viocr/selected_images_new',help="Directory storing images")
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument('--filter', nargs='+',default=["1","2","3","4","5","6","7","32","33","34","35","36","38","39","40","41"], help="low vision filter id")
    parser.add_argument('--acuity_file', type=str,default='data/human/SelectedFilter.csv')
    parser.add_argument('--text_desc', action='store_true',help="Use text description to describe the image")
    
    args = parser.parse_args()
    model_name = 'qwen2.5'
    # concept_file = args.infile.split('/')[-1]
    outfile_jsonl = os.path.join(args.outfile,f'{args.model_path.split("/")[-1]}{"_text_desc" if args.text_desc else ""}.jsonl')
    output_file = os.path.join(args.outfile,f'{args.model_path.split("/")[-1]}{"_text_desc" if args.text_desc else ""}.json')
    os.makedirs(args.outfile, exist_ok=True)

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True,
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-3B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Specify hyperparameters for generation
    # model.generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)
    
    samples = json.load(open(args.infile, "r"))['images']
    output_samples = []
    if args.text_desc:
        import pandas as pd
        acuity_df = pd.read_csv(args.acuity_file)
        acuity_df = acuity_df[['Filter_no', 'VA', 'CS']]
        acuity_dict = acuity_df.set_index('Filter_no').to_dict(orient='index')
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
                if args.text_desc:
                    text_desc = f"You are a low vision reader. Your visual acuity is {acuity_dict[int(formatted_sample['filter'])]['VA']} logMAR and your contrast sensitivity is {acuity_dict[int(formatted_sample['filter'])]['CS']} logCS."
                    q = text_desc + '\n' + q
                    image_file = os.path.join(args.img_dir,'16', sample["file_name"].split("/")[-1])
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
                            "text": q},
                        ],
                    }
                ]
                output = eval_model(processor, model, messages)
                output = output.strip()
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