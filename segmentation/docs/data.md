# Data Preparation

Our setup follows [Referformer](https://github.com/wjn922/ReferFormer)
Create a new directory `data` to store all the datasets.

## 🖼️ Ref-COCO

Download the dataset from the official website [COCO](https://cocodataset.org/#download).   
RefCOCO/+/g use the ```COCO2014``` train split.
Download the annotation files from [github](https://github.com/lichengunc/refer).

Convert the annotation files:

```
python3 tools/data/convert_refexp_to_coco.py
```

Finally, we expect the directory structure to be the following:

```
SAMWISE
├── data
│   ├── coco
│   │   ├── train2014
│   │   ├── refcoco
│   │   │   ├── instances_refcoco_train.json
│   │   │   ├── instances_refcoco_val.json
│   │   ├── refcoco+
│   │   │   ├── instances_refcoco+_train.json
│   │   │   ├── instances_refcoco+_val.json
│   │   ├── refcocog
│   │   │   ├── instances_refcocog_train.json
│   │   │   ├── instances_refcocog_val.json
```


## 🎥 Ref-Youtube-VOS

Download the dataset from the competition's website [here](https://competitions.codalab.org/competitions/29139#participate-get_data).
Then, extract and organize the file. We expect the directory structure to be the following:

```
SAMWISE
├── data
│   ├── ref-youtube-vos
│   │   ├── meta_expressions
│   │   ├── train
│   │   │   ├── JPEGImages
│   │   │   ├── Annotations
│   │   │   ├── meta.json
│   │   ├── valid
│   │   │   ├── JPEGImages
```

## 🎬 Ref-DAVIS17

Downlaod the DAVIS2017 dataset from the [website](https://davischallenge.org/davis2017/code.html). Note that you only need to download the two zip files `DAVIS-2017-Unsupervised-trainval-480p.zip` and `DAVIS-2017_semantics-480p.zip`.
Download the text annotations from the [website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/video-segmentation/video-object-segmentation-with-language-referring-expressions).
Then, put the zip files in the directory as follows.


```
SAMWISE
├── data
│   ├── ref-davis
│   │   ├── DAVIS-2017_semantics-480p.zip
│   │   ├── DAVIS-2017-Unsupervised-trainval-480p.zip
│   │   ├── davis_text_annotations.zip
```

Unzip these zip files.
```
unzip -o davis_text_annotations.zip
unzip -o DAVIS-2017_semantics-480p.zip
unzip -o DAVIS-2017-Unsupervised-trainval-480p.zip
```

Preprocess the dataset to Ref-Youtube-VOS format. (Make sure you are in the main directory)

```
python tools/data/convert_davis_to_ytvos.py
```

Finally, unzip the file `DAVIS-2017-Unsupervised-trainval-480p.zip` again (since we use `mv` in preprocess for efficiency).

```
unzip -o DAVIS-2017-Unsupervised-trainval-480p.zip
```

## 🐦 MeViS

Download and unzip the [dataset](https://codalab.lisn.upsaclay.fr/competitions/15094).
```
unzip -o MeViS_release.zip
```

The dataset follows a similar structure as Refer-Youtube-VOS. 
Each split of the dataset consists of three parts: 
```JPEGImages```, which holds the frame images, ```meta_expressions.json```,
which provides referring expressions and metadata of videos, 
and ```mask_dict.json```, which contains the ground-truth masks of objects.
Ground-truth segmentation masks are saved in the format of COCO RLE,
and expressions are organized similarly like Refer-Youtube-VOS.


```
SAMWISE
├── data
│   ├── MeViS_release
│   │   ├── train
│   │   │   ├──JPEGImages
│   │   │   ├──mask_dict.json
│   │   │   ├──meta_expressions.json
│   │   ├── valid_u
│   │   │   ├──JPEGImages
│   │   │   ├──mask_dict.json
│   │   │   ├──meta_expressions.json
│   │   ├── valid
│   │   │   ├──JPEGImages
│   │   │   ├──mask_dict.json
│   │   │   ├──meta_expressions.json
```



