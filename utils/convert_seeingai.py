#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd


def _split_cell(cell: Any, lowercase: bool = False, dedup: bool = True) -> List[str]:
    """Split a cell by commas, remove leading/trailing spaces and empty items, optionally lowercase and deduplicate."""
    if pd.isna(cell):
        return []
    tokens = [t.strip() for t in str(cell).split(',')]
    tokens = [t for t in tokens if t]  # 去空
    if lowercase:
        tokens = [t.lower() for t in tokens]
    if dedup:
        tokens = list(dict.fromkeys(tokens))  # 保序去重
    return tokens


def _to_int_if_possible(x: Union[str, int]) -> Union[int, str]:
    """If the header is a numeric string, convert it to int; otherwise, keep it as is."""
    try:
        return int(str(x).strip())
    except Exception:
        return str(x).strip()


def excel_tab_to_records(
    image_id_map: Dict[str, int],
    xlsx_path: str,
    sheet_name: str,
    *,
    include_empty: bool = True,
    lowercase: bool = False,
    dedup: bool = True,
    category_id: int = 1,
    rec_score: float = 0.0,
    det_score: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Convert a specific sheet of an Excel file to a JSON array format.
    The first column: image name (used to group and generate image_id in order)
    The subsequent columns: various filters; cells are split by English commas.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=0)

    if df.shape[1] < 2:
        raise ValueError("工作表需要至少两列：第一列为图片名，其余为 filter 列。")

    # 生成 filter id（来自表头；为空则回退到列序号 1 开始）
    filter_ids: List[Union[int, str]] = []
    for idx, col in enumerate(df.columns[1:], start=1):
        name = str(col).strip()
        if not name or name.lower() in {"nan", "none"}:
            filter_ids.append(idx)
        else:
            filter_ids.append(_to_int_if_possible(name))

    records: List[Dict[str, Any]] = []

    # image_id 按行顺序编号
    # image_id_map: Dict[str, int] = {}
    next_image_id = 0

    for _, row in df.iterrows():
        img = str(row.iloc[0]).strip()
        if not img or img.lower().startswith("nan"):
            # 跳过图片名为空的行
            continue

        # if img not in image_id_map:
        #     image_id_map[img] = next_image_id
        #     next_image_id += 1
        # img_id = image_id_map[img]

        for c_idx, fval in enumerate(filter_ids, start=1):
            img_id = image_id_map[f'{fval}/{img if str.endswith(img, ".jpg") else img + ".jpg"}']
            cell = row.iloc[c_idx] if c_idx < len(row) else ""
            tokens = _split_cell(cell, lowercase=lowercase, dedup=dedup)

            if not tokens and not include_empty:
                continue

            rec_texts_str = str(tokens)  # 关键：生成 "['A', 'B']" 这种字符串

            records.append({
                "image_id": img_id,
                "category_id": category_id,
                "polys": [],
                "rec_texts": rec_texts_str,
                "rec_score": rec_score,
                "det_score": det_score,
                "filter": fval,
            })

    return records


def save_records(records: List[Dict[str, Any]], output_json: str) -> None:
    Path(output_json).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON saved to: {Path(output_json).resolve()}")


if __name__ == "__main__":
    import argparse
    # python xlsx_tab_to_json_records.py
    parser = argparse.ArgumentParser(description="Convert a specific sheet of an Excel file to a JSON array format")
    parser.add_argument("--xlsx_path", default="/cis/home/qgao14/my_documents/VIOCR_infer_models/data/summary/SeeingAI.xlsx", help="input .xlsx file path")
    parser.add_argument("--sheet_name", default="SeeingAI-totaltext", help="sheet/tab")
    parser.add_argument("-o", "--output", default="output/totaltext/seeingai.json", help="output JSON path (optional)")
    parser.add_argument("--lowercase", action="store_true", help="convert tokens to lowercase")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false", help="do not deduplicate tokens within cells")
    parser.add_argument("--no-empty", dest="include_empty", action="store_false", help="do not output empty records")
    parser.add_argument("--category-id", type=int, default=1, help="category_id (default: 1)")
    parser.add_argument("--rec-score", type=float, default=0.0, help="rec_score (default: 0)")
    parser.add_argument("--det-score", type=float, default=0.0, help="det_score (default: 0)")
    parser.add_argument("--anno_path", default="data/totaltext/anno.json", help="annotation JSON path")

    args = parser.parse_args()
    with open(args.anno_path, 'r') as f:
        anno = json.load(f)
    image_id_map = anno['images']
    image_id_map = {item['file_name']: item['id'] for item in image_id_map}

    recs = excel_tab_to_records(
        image_id_map,
        args.xlsx_path,
        args.sheet_name,
        include_empty=args.include_empty,
        lowercase=args.lowercase,
        dedup=args.dedup,
        category_id=args.category_id,
        rec_score=args.rec_score,
        det_score=args.det_score
    )
    save_records(recs, args.output)
