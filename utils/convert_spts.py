from glob import glob
import json
import os


anno_path = 'data/totaltext/anno.json'
with open(anno_path, 'r') as f:
    anno = json.load(f)
id_dict = {item['file_name']:item['id'] for item in anno['images']}
results = []
input_dir = '/cis/home/qgao14/my_documents/VIOCR_infer_models/model_zoo/output/totaltext/maerec/'
for file in glob(input_dir+'*/*/*.json'):
    
    with open(file, 'r') as f:
        data = json.load(f)
    file = file.replace('preds/', '')
    filter_no, img_name = file.split('/')[-2:]
    img_name = img_name.split('.')[0]
    filename = f'{filter_no}/{img_name}.jpg'
    data = [data] if isinstance(data, dict) else data
    for item in data:
        res = {}
        res['image_id'] = id_dict[filename]
        res['category_id'] = 1
        res['rec_texts'] = str(item['rec_texts'])
        res['polys'] = item['boxes'] if 'boxes' in item else item['det_polygons']
        res['rec_scores'] = item.get('rec_scores', [0])
        res['det_score'] = item.get('det_scores', [0])
        res['filter'] = int(filter_no)
        results.append(res)

with open('output/totaltext/maerec.json', 'w') as f:
    json.dump(results, f, indent=4)

    