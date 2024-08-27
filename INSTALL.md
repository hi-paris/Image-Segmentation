## 01 📦 Installation

### download Miniconda installer (from conda-forge)

wget [https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh) \
-O [miniconda.sh](http://miniconda.sh/)

### install Miniconda

MINICONDA_PATH=./miniconda3
chmod +x [miniconda.sh](http://miniconda.sh/) && ./miniconda.sh -b -p $MINICONDA_PATH

### make sure conda is up-to-date

source $MINICONDA_PATH/etc/profile.d/conda.sh
conda update --yes conda

### Update your .bashrc to initialise your conda base environment on each login

conda init

```
conda create --name fcclip python=3.8 -y
conda activate fcclip
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
git clone https://github.com/bytedance/fc-clip.git
cd fcclip
pip install -r requirements.txt
cd fcclip/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
pip install open-clip-torch==2.24.0
```

/home/infres/gbrison/segmentation/benchmark-mllm/datasets/cityscapes

### How to download the Cityscapes dataset

In the first command, put your username and password. This will login with your credentials and keep the associated cookie:

```
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=myusername&password=mypassword&submit=Login' https://www.cityscapes-dataset.com/login/
```

In the second command, you need to provide the packageID parameter:

```
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
```

For example, if you want to download “gtFine_trainvaltest.zip” (241MB), you need to set “packageID=1”.

And to download “leftImg8bit_trainvaltest.zip” (11GB), you need to set “packageID=3”.

Once you download the zip files that you need, you can unzip them using the command:

```
unzip gtFine_trainvaltest.zip

unzip leftImg8bit_trainvaltest.zip
```

### Dataset preparation:

The goal here is to get the following structure for the cityscapes dataset:

```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # below are generated Cityscapes panoptic annotation
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
```

Set location for the datasets folder:

```
export DETECTRON2_DATASETS=/path/to/datasets
```

Install cityscapes scripts by:

```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Note: to create labelTrainIds.png, first prepare the above structure, then run cityscapesescript with:

```
export CITYSCAPES_DATASET=/home/infres/gbrison/fcclip_new/fc-clip/datasets/cityscapes 

python /home/infres/gbrison/fcclip_new/fc-clip/cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

These files are not needed for instance segmentation.

Note: to generate Cityscapes panoptic dataset, run cityscapesescript with:

```
export CITYSCAPES_DATASET=/home/infres/gbrison/fcclip_new/fc-clip/datasets/cityscapes 

python /home/infres/gbrison/fcclip_new/fc-clip/cityscapesScripts/cityscapesscripts/preparation/createPanopticImgs.py
```

These files are not needed for semantic and instance segmentation.

/home/infres/gbrison/fcclip_new/fc-clip/datasets/cityscapes

Run the code :

python train_net_GB.py --config-file config-GB_001.yaml --num-gpus 2

/home/infres/gbrison/fcclip_new/fc-clip/datasets

### How to register and use a custom dataset

Follow these instructions on how to use Detectron2 dataset APIs to add and use custom datasets: 

[https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)

This an example of how the Cityscapes panoptic dataset (used in FC-CLIP) is registered to the DatasetCatalog :

⚠️ In order to get access to the different parameters (eg: CITYSCAPES_CATEGORIES), you need to check the file directly in the FC-CLIP package: 
 
Path to the file: **fc-clip/fcclip/data/datasets/register_cityscapes_panoptic.py**

```python
"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/cityscapes_panoptic.py
"""

import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from . import openseg_classes

CITYSCAPES_CATEGORIES = openseg_classes.get_cityscapes_categories_with_prompt_eng()

"""
This file contains functions to register the Cityscapes panoptic dataset to the DatasetCatalog.
"""

logger = logging.getLogger(__name__)

def get_cityscapes_panoptic_files(image_dir, gt_dir, json_info):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    image_dict = {}
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            suffix = "_leftImg8bit.png"
            assert basename.endswith(suffix), basename
            basename = os.path.basename(basename)[: -len(suffix)]

            image_dict[basename] = image_file

    for ann in json_info["annotations"]:
        image_file = image_dict.get(ann["image_id"], None)
        assert image_file is not None, "No image {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = ann["segments_info"]

        files.append((image_file, label_file, segments_info))

    assert len(files), "No images found in {}".format(image_dir)
    assert PathManager.isfile(files[0][0]), files[0][0]
    assert PathManager.isfile(files[0][1]), files[0][1]
    return files

def load_cityscapes_panoptic(image_dir, gt_dir, gt_json, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train".
        gt_json (str): path to the json file. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train.json".
        meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
            and "stuff_dataset_id_to_contiguous_id" to map category ids to
            contiguous ids for training.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        return segment_info

    assert os.path.exists(
        gt_json
    ), "Please run `python cityscapesscripts/preparation/createPanopticImgs.py` to generate label files."  # noqa
    with open(gt_json) as f:
        json_info = json.load(f)
    files = get_cityscapes_panoptic_files(image_dir, gt_dir, json_info)
    ret = []
    for image_file, label_file, segments_info in files:
        sem_label_file = (
            image_file.replace("leftImg8bit", "gtFine").split(".")[0] + "_labelTrainIds.png"
        )
        segments_info = [_convert_category_id(x, meta) for x in segments_info]
        ret.append(
            {
                "file_name": image_file,
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(image_file))[0].split("_")[:3]
                ),
                "sem_seg_file_name": sem_label_file,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    assert PathManager.isfile(
        ret[0]["pan_seg_file_name"]
    ), "Please generate panoptic annotation with python cityscapesscripts/preparation/createPanopticImgs.py"  # noqa
    return ret

# rename to avoid conflict
_RAW_CITYSCAPES_PANOPTIC_SPLITS = {
    "openvocab_cityscapes_fine_panoptic_train": (
        "cityscapes/leftImg8bit/train",
        "cityscapes/gtFine/cityscapes_panoptic_train",
        "cityscapes/gtFine/cityscapes_panoptic_train.json",
    ),
    "openvocab_cityscapes_fine_panoptic_val": (
        "cityscapes/leftImg8bit/val",
        "cityscapes/gtFine/cityscapes_panoptic_val",
        "cityscapes/gtFine/cityscapes_panoptic_val.json",
    ),
    # "cityscapes_fine_panoptic_test": not supported yet
}

def register_all_cityscapes_panoptic(root):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in CITYSCAPES_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (image_dir, gt_dir, gt_json) in _RAW_CITYSCAPES_PANOPTIC_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        DatasetCatalog.register(
            key, lambda x=image_dir, y=gt_dir, z=gt_json: load_cityscapes_panoptic(x, y, z, meta)
        )
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            gt_dir=gt_dir.replace("cityscapes_panoptic_", ""),
            evaluator_type="cityscapes_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_cityscapes_panoptic(_root)
```
