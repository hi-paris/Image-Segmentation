# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from PIL import Image
from tabulate import tabulate
import multiprocessing
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from panopticapi.utils import get_traceback, rgb2id
import json
from .evaluator import DatasetEvaluator
import cv2
logger = logging.getLogger(__name__)

class COCOPanopticEvaluator(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }


        
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb 
        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = [] 
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f) 


            
            json_data["annotations"] = self._predictions
            

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))
            
            
            from panopticapi.evaluation import pq_compute, pq_compute_multi_core,pq_compute_single_core
            
            
            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )
                #_print_confusion_matrix_of_performance(gt_json, PathManager.get_local_path(predictions_json) , gt_folder, pred_dir)
            
 
        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]
        for i in pq_res["per_class"]:
            res["PQ_"+str(i)] = 100 * pq_res["per_class"][i]["pq"]
            res["SQ_"+str(i)] = 100 * pq_res["per_class"][i]["sq"]
            res["RQ_"+str(i)] = 100 * pq_res["per_class"][i]["rq"]
        

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results





def save_result_json(path, pq_res):
    fichier_json = path

    try: 
        with open(fichier_json, 'r') as f:
            data = json.load(f)
    except FileNotFoundError: 
        data = {}

    if len(data)>0:
        id=list(data.keys())[-1]
    else:
        id=0
    id=int(id)+1
    data[id]=str(pq_res) 

    with open(fichier_json, 'w') as f:
        json.dump(data, f, indent=4)




OFFSET = 256 * 256 * 256
VOID = 0
def save_json(list_data_item,list_paths,get_ann_list):
    
 
    fichier_json = "Results.json"

    try: 
        with open(fichier_json, 'r') as f:
            data = json.load(f)
    except FileNotFoundError: 
        data = {}

    if len(data)>0:
        id=list(data.keys())[-1]
    else:
        id=-1
    id=int(id)
    for data_item in list_data_item:
        id+=1
        data[id]=str(data_item)
        
    
    with open(fichier_json, 'w') as f:
        json.dump(data, f, indent=4) 
    
    fichier_json = "True_labels.json"

    try: 
        with open(fichier_json, 'r') as f:
            data = json.load(f)
    except FileNotFoundError: 
        data = {}

    if len(data)>0:
        id=list(data.keys())[-1]
    else:
        id=-1
    id=int(id)
    for data_item in get_ann_list:
        id+=1
        data[id]=str(data_item)
        
    
    with open(fichier_json, 'w') as f:
        json.dump(data, f, indent=4) 

    fichier_json = "paths.json"

    try: 
        with open(fichier_json, 'r') as f:
            data = json.load(f)
    except FileNotFoundError: 
        data = {}

    if len(data)>0:
        id=list(data.keys())[-1]
    else:
        id=-1
    id=int(id)
    mypath = str(int(id/499))
    if not os.path.isdir(mypath):
        os.makedirs(f'/home/ids/gbrison/FC/fc-clip/images/{mypath}')
    for data_item in list_paths:
        im=cv2.imread(str(data_item[1]))
        path_des=''.join(data_item[1].split('/'))
        cv2.imwrite(f'/home/ids/gbrison/FC/fc-clip/images/{mypath}/{path_des}', im)
        
        id+=1
        data[id]=str([data_item[0], f'/{mypath}/{path_des}', data_item[2]])
        
    
    with open(fichier_json, 'w') as f:
        json.dump(data, f, indent=4)  



def _print_confusion_matrix_of_performance(gt_json, PathManag , gt_folder, pred_dir):
    gt_json_file=gt_json
    pred_json_file=PathManag  
    gt_folder=gt_folder
    pred_folder=pred_dir

    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)
    logger.info("Start of intermediate result printing")
    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    
    categories = {el['id']: el for el in gt_json['categories']}
    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    #logger.info("pred_annotations : "+str(pred_annotations))
    #logger.info("categories : "+str(categories))





    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id'] 
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))
    #logger.info("matched_annotations_list : "+str(matched_annotations_list))
    
    
    list_json=[]
    list_paths=[]
    get_ann_list=[]
    for gt_ann, pred_ann  in matched_annotations_list:
        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)
        
        #logger.info("===============================================================\n"  )
        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        #logger.info("pred_labels_set ______ :\n" + str(pred_labels_set))
        #logger.info("pred_segms ______ :\n" + str(pred_segms))
        #logger.info("gt_segms ______ :\n" + str(gt_segms))
        #logger.info("gt_segms len ______ :\n" + str(len(gt_segms)))
        #logger.info("gt_ann['segments_info'] ______ :\n" + str(gt_ann['segments_info']))
        #logger.info("pred_ann['segments_info'] ______ :\n" + str(pred_ann['segments_info']))
        #logger.info("pred_ann['segments_info'] ______ :\n" + str(pred_ann['segments_info']))
        labels, labels_cnt = np.unique(pred_ann, return_counts=True) 
        #logger.info("predected labels ______ :\n" + str(labels))
        #logger.info("predected labels len ______ :\n" + str(len(labels[0]['segments_info'])))
        



        
        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        
        #logger.info("predected labels with confusion matrix len ______ :\n" + str(len(labels)))
        #logger.info("predected labels with confusion matrix ______ :\n" + str(labels))

        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection
        #logger.info("gt_pred_map : "+str(len(gt_pred_map)))
        
         

        max_values = {}
        index_val={}
        result_glb={}


        for key, val in gt_pred_map.items():
            if val>0.5:
                a, b = key
                if a in max_values:
                    if val > max_values[a]:
                        max_values[a] = val
                        index_val[a]=b
                else:
                    max_values[a] = val
                    index_val[a]=b

        for a in index_val:
            if (a,index_val[a]) not in result_glb:
                result_glb[(a,index_val[a])]=1
            else:
                result_glb[(a,index_val[a])]+=1
        #logger.info("result_glb ______ :\n" + str(result_glb))
        #logger.info("result_glb len ______ :\n" + str(len(result_glb)))
        list_paths.append([gt_folder+"/"+gt_ann['file_name'],pred_folder+"/"+ pred_ann['file_name'],gt_ann['file_name']])
        get_ann_list.append(gt_ann['segments_info'])
        #logger.info("image_____ s "+pred_folder+"/"+ pred_ann['file_name'])
        list_json.append(result_glb)
        
    save_json(list_json,list_paths, get_ann_list)
        

    #logger.info("result ______ :\n" + str(result_glb))
    


def _print_panoptic_results(pq_res):
    save_result_json("json_results.json", pq_res) 
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)

    lp=[24,25,26,27,28,31,32,33]

    data = []
    for name in pq_res["per_class"]: 
            if name in lp:
                row = ["class_"+str(name)] + [pq_res["per_class"][name][k] * 100 for k in ["pq", "sq", "rq"]] + ["Things"]
            else:
                row = ["class_"+str(name)] + [pq_res["per_class"][name][k] * 100 for k in ["pq", "sq", "rq"]] + ["Stuff"]
            data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )
        _print_panoptic_results(pq_res)
