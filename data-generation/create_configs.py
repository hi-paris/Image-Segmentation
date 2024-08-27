import os
import pickle
import json
import copy
import sys

def read_files(path_found_matcher, path, path_prediction):
    """
    Reads the necessary input files.

    Parameters
    ----------
    path_found_matcher : str
        The path to the pickle file containing matcher data.
    path : str
        The path to the original JSON file.
    path_prediction : str
        The path to the prediction JSON file.

    Returns
    -------
    tuple
        A tuple containing the loaded matcher data, original JSON, and prediction JSON.
    """
    with open(path_found_matcher, 'rb') as file:
        matcher_pickle = pickle.load(file) 
    with open(path, 'r') as file:
        original_json = json.load(file)
    with open(path_prediction, 'r') as file:
        prediction = json.load(file)

    return matcher_pickle, original_json, prediction

def get_matcher(data_for_matcher, index):
    """
    Creates a matcher dictionary from the given data for a specific index.

    Parameters
    ----------
    data_for_matcher : dict
        The data containing matching information.
    index : int
        The index of the data to process.

    Returns
    -------
    dict
        A dictionary mapping the matched IDs.
    """
    max_per_y = {}
    matcher = {}
    data_ = data_for_matcher[index]
    for (x, y), value in data_.items():
        if y not in max_per_y or value > max_per_y[y][1]:
            max_per_y[y] = (x, value)
            if len(str(x)) > 3:
                matcher[int(y)] = int(str(x)[:2])
            else:
                matcher[int(y)] = int(x)
    return matcher

def create_invocab_json(in_vocab_json, list_of_matcher, prediction, out_vocab_labels):
    """
    Modifies the in_vocab_json based on the matcher and prediction data to create an Invocab configuration.

    Parameters
    ----------
    in_vocab_json : dict
        The JSON data to modify.
    list_of_matcher : list
        The list of matchers for all images.
    prediction : dict
        The prediction JSON data.
    out_vocab_labels : list
        The list of labels considered out-of-vocabulary.

    Returns
    -------
    dict
        The modified JSON data representing the Invocab configuration.
    """
    for annot_new, matcher, prediction in zip(in_vocab_json["annotations"], list_of_matcher, prediction['annotations']):
        for segment in annot_new["segments_info"]:
            if segment['category_id'] in out_vocab_labels:
                if segment['id'] in matcher:
                    for pred_segme in prediction['segments_info']:
                        if matcher[segment['id']] == pred_segme['id']:
                            segment['category_id'] = pred_segme["category_id"]
                            break
    return in_vocab_json

def create_normal_json(original_json, in_vocab_labels):
    """
    Creates a Normal configuration JSON from the original JSON by filtering out certain labels.

    Parameters
    ----------
    original_json : dict
        The original JSON data.
    in_vocab_labels : list
        The list of labels considered in-vocabulary.

    Returns
    -------
    dict
        The modified JSON data representing the Normal configuration.
    """
    normal_json = copy.deepcopy(original_json)
    for anno in range(len(normal_json["annotations"])):
        list_annotations = []
        for i in range(len(normal_json["annotations"][anno]["segments_info"])):
            if normal_json["annotations"][anno]["segments_info"][i]['category_id'] in in_vocab_labels:
                list_annotations.append(normal_json["annotations"][anno]["segments_info"][i])
        normal_json["annotations"][anno]["segments_info"] = list_annotations
    return normal_json

def save_json(data, path):
    """
    Saves the provided data as a JSON file to the specified path.

    Parameters
    ----------
    data : dict
        The JSON data to save.
    path : str
        The file path where the JSON data will be saved.

    Returns
    -------
    None
    """
    with open(path, 'w') as json_file:
        json.dump(data, json_file)

def main():
    """
    Main function to execute the script based on the provided mode.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    if len(sys.argv) != 2:
        print("Usage: python configs.py <Invocab|Normal|Naive>")
        sys.exit(1)

    mode = sys.argv[1]

    # Config paths
    path_results = ""
    path_prediction = "predictions.json"
    path = "cityscapes_panoptic_train.json"
    path_found_matcher = 'all1'

    # Read input files
    matcher_pickle, original_json, prediction = read_files(path_found_matcher, path, path_prediction)

    # Filter labels
    list_of_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    conf = 8
    in_vocab_labels = list_of_labels[:-conf]
    out_vocab_labels = list_of_labels[-conf:]

    # Create matchers for all images
    list_of_matcher = [get_matcher(matcher_pickle, i) for i in range(len(original_json['annotations']))]

    # Prepare JSONs for different modes
    if mode == "Invocab":
        in_vocab_json = copy.deepcopy(original_json)
        in_vocab_json = create_invocab_json(in_vocab_json, list_of_matcher, prediction, out_vocab_labels)
        path_in_vocab_008 = os.path.join(path_results, "cityscapes_panoptic_train_invocab_008.json")
        save_json(in_vocab_json, path_in_vocab_008)

    elif mode == "Normal" or mode == "Naive":
        normal_json = create_normal_json(original_json, in_vocab_labels)
        path_normal_008 = os.path.join(path_results, "cityscapes_panoptic_train_normal_008.json")
        save_json(normal_json, path_normal_008)

        if mode == "Naive":
            naive_json = copy.deepcopy(normal_json)
            path_naive_008 = os.path.join(path_results, "cityscapes_panoptic_train_naive_008.json")
            save_json(naive_json, path_naive_008)

    else:
        print("Invalid mode provided. Use 'Invocab', 'Normal', or 'Naive'.")
        sys.exit(1)

    print("Results are saved in the folder:", path_results)

if __name__ == "__main__":
    main()
