import os
import pickle
import json
import copy


# Config paths 
path_results =  "/tsi/hi-paris/GB/segmentation/Inference_train_data_new/cl_inference_19"
path_prediction=path_results+"/inference/predictions.json"
path="/home/ids/gbrison/segmentation/segmentation/fc-clip/datasets/cityscapes/gtFine/cityscapes_panoptic_train.json"
path_found_matcher=os.path.join(path_results,'all', 'all'+str(1))
# choose your config
conf = 8

# Read files
with open(path_found_matcher, 'rb') as file:
    matcher_pickle=pickle.load(file) 
with open(path, 'r') as file:
    original_json = json.load(file)
with open(path_prediction, 'r') as file:
    prediction = json.load(file)

# filter labels
list_of_labels= [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]



in_vocab_labels=list_of_labels[:-conf]
out_vocab_labels=list_of_labels[-conf:]



# get the matching of ids of segment
def get_matcher(data_for_matcher, index):
    max_per_y = {}
    matcher={}
    data_=data_for_matcher[index] 
    for (x, y), value in data_.items():
        if y not in max_per_y or value > max_per_y[y][1]:
            max_per_y[y] = (x, value)
            if len(str(x))>3:
                matcher[int(y)]=int(str(x)[:2])
            else:
                matcher[int(y)]=int(x)
    return matcher

# create matchers for all images
list_of_matcher= [ get_matcher(matcher_pickle, i) for i in range(len(original_json['annotations']))]


# Prepare the jsons of results
normal_json = copy.deepcopy(original_json)
in_vocab_json = copy.deepcopy(original_json)

# Create Invocab config
for annot_new, matcher, prediction in zip(in_vocab_json["annotations"],list_of_matcher, prediction['annotations']):
        for segment in annot_new["segments_info"]:
            if segment['category_id'] in out_vocab_labels:
                if segment['id'] in matcher:
                    for pred_segme in prediction['segments_info']:
                        if matcher[segment['id']]  == pred_segme['id']:
                            segment['category_id']=pred_segme["category_id"]
                            break  

#Create Normal and Naive jsons
for anno in range(len(normal_json["annotations"])):
        list_annotations=[]
        for i in range(len(normal_json["annotations"][anno]["segments_info"])):
            if normal_json["annotations"][anno]["segments_info"][i]['category_id'] in in_vocab_labels:
                list_annotations.append(normal_json["annotations"][anno]["segments_info"][i])
        normal_json["annotations"][anno]["segments_info"]= list_annotations

# nave use the same input of normal with modefication of the output
naive_json =  copy.deepcopy(normal_json)

# prepare paths of results
path_naive_008=os.path.join(".","cityscapes_panoptic_train_naive_008_cl.json")
path_normal_008=os.path.join(".","cityscapes_panoptic_train_normal_008_cl.json")
path_in_vocab_008=os.path.join(".","cityscapes_panoptic_train_invocab_008_cl.json")

# save results
with open(path_naive_008, 'w') as json_file:
    json.dump(naive_json, json_file)
with open(path_normal_008, 'w') as json_file:
    json.dump(normal_json, json_file)
with open(path_in_vocab_008, 'w') as json_file:
    json.dump(in_vocab_json, json_file)


print("Results are saved in the folder: ", path_results)
