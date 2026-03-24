# 🛠️ Training and Inference Guide  

This guide provides step-by-step instructions for training and evaluating **SAMWISE** on various datasets.

---

## 📥 Pre-trained Weights  

First, download the pre-trained model:  

```
cd pretrain/
gdown --fuzzy https://drive.google.com/file/d/1gRGzARDjIisZ3PnCW77Y9TMM_SbV8aaa/view?usp=drive_link
```
Ensure the pre-trained model is correctly stored in the ```pretrain/``` directory before proceeding.

## 🎥 Training & Inference for Ref-Youtube-VOS
### Training on Ref-Youtube-VOS
Run the following command to train SAMWISE on Ref-Youtube-VOS:
```
python3 main.py --resume=pretrain/pretrained_model.pth --dataset_file ytvos --HSA --use_cme_head --name_exp [name_exp] --epochs 4 --batch_size 4
```
### Inference on Ref-Youtube-VOS
To perform inference using the trained model:

```
python3 inference_ytvos.py --resume=[/path/to/model_weight] --name_exp [name_exp] --HSA --use_cme_head
```

### Submitting Results
To prepare the results for submission:
```
cd output/[exp_name]
zip -qq -r Annotations.zip Annotations
```
Upload ```Annotations.zip``` to the [Ref-Youtube-VOS Competition Server.](https://codalab.lisn.upsaclay.fr/competitions/3282)

## 🎬 Inference for Ref-DAVIS

We report the results using the model trained on Ref-Youtube-VOS without finetune.

Run inference with: 
```
python3 inference_davis.py --resume=[/path/to/model_weight] --name_exp [name_exp] --HSA --use_cme_head
```

## 🐦 Training & Inference for MeViS
### Training on MeViS

Run the following command to train SAMWISE on MeViS:
```
python3 main.py --resume=pretrain/pretrained_model.pth --dataset_file mevis --HSA --use_cme_head --name_exp [name_exp] --epochs 1 --batch_size 4 
```
### Inference on MeViS
To run inference on the trained model:

```
python3 inference_mevis.py --resume=[/path/to/model_weight] --name_exp [name_exp] --HSA --use_cme_head
```
### Submitting Results

Prepare the results for submission:
```
cd output/[exp_name]/Annotations
zip -qq -r ../Annotations.zip *
cd ..
```
Upload ```Annotations.zip``` to the [MeViS Competition Server](https://codalab.lisn.upsaclay.fr/competitions/21944).

### Local Evaluation on MeViS 
To evaluate locally on the ```valid_u split```, run:

```
python3 inference_mevis.py --split valid_u --resume=[/path/to/model_weight] --name_exp [name_exp] --HSA --use_cme_head
```

## 🖼️ Pre-training
Before fine-tuning on downstream datasets, 
we pre-train SAMWISE on **RefCOCO**, **RefCOCO+**, and **RefCOCOg** datasets ([here the weights](https://drive.google.com/file/d/1gRGzARDjIisZ3PnCW77Y9TMM_SbV8aaa/view?usp=drive_link)).

To replicate the pre-training process, run:
```
python main_pretrain.py --lr 1e-4 --clip_max_norm 0.1 --batch_size 32 --num_frames 1 --name_exp [name_exp] --dataset_file all --epochs 6 --disable_pred_obj_score 
```

Once pre-trained, you can evaluate the model on a Referring Image Segmentation (RIS) benchmark using:

```
python3 main_pretrain.py --eval --resume=[/path/to/model_weight] --name_exp [name_exp] --disable_pred_obj_score 
```


## ⚡ Multi-GPU Training

All training scripts in this repository support multi-GPU training using `torch.distributed`.
To enable distributed training, prepend the training command with:

```
python -m torch.distributed.launch --nproc_per_node=N_GPUS --master_port=FREE_PORT --use_env 
```

Replace N_GPUS with the desired number of GPUS and FREE_PORT with an available port on your machine (e.g., 1234).

For example, to train on Ref-YouTube-VOS using 4 GPUs:


```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 --use_env main.py --resume=pretrain/pretrained_model.pth --dataset_file ytvos --HSA --use_cme_head --name_exp [name_exp] --epochs 1 --batch_size 4
```

Make sure to adjust --batch_size appropriately based on your available GPU memory.
