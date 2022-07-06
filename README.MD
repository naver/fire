# Learning Super-Features for Image Retrieval


This repository contains the code for running our FIRe model presented in our [ICLR'22 paper](https://openreview.net/pdf?id=wogsFPHwftY):

```bibtex
@inproceedings{superfeatures,
  title={{Learning Super-Features for Image Retrieval}},
  author={{Weinzaepfel, Philippe and Lucas, Thomas and Larlus, Diane and Kalantidis, Yannis}},
  booktitle={{ICLR}},
  year={2022}
}
```

## License

The code is distributed under the CC BY-NC-SA 4.0 License. See [LICENSE](LICENSE) for more information.
It is based on code from [HOW](https://github.com/gtolias/how), [cirtorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch/) and [ASMK](https://github.com/jenicek/asmk) that are released under their own license, the MIT license.

## Preparation

After cloning this repository, you must also have HOW, cirtorch and ASMK and have them in your PYTHONPATH.

1. install [HOW](https://github.com/gtolias/how)
```
git clone https://github.com/gtolias/how
export PYTHONPATH=${PYTHONPATH}:$(realpath how)
```

2. install [cirtorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch/)
```
wget "https://github.com/filipradenovic/cnnimageretrieval-pytorch/archive/v1.2.zip"
unzip v1.2.zip
rm v1.2.zip
export PYTHONPATH=${PYTHONPATH}:$(realpath cnnimageretrieval-pytorch-1.2)
```

3. install [ASMK](https://github.com/jenicek/asmk)
```
git clone https://github.com/jenicek/asmk.git
pip3 install pyaml numpy faiss-gpu
cd asmk
python3 setup.py build_ext --inplace
rm -r build
cd ..
export PYTHONPATH=${PYTHONPATH}:$(realpath asmk)
```

4. install dependencies by running:
```
pip3 install -r how/requirements.txt
```

5. data/experiments folders

All data will be stored under a folder ``fire_data`` that will be created when running the code; similarly, results and models from all experiments will be stored under folder ``fire_experiments``

## Evaluating our ICLR'22 FIRe model

To evaluate on ROxford/RParis our model trained on SfM-120k, simply run
```
python evaluate.py eval_fire.yml
```

With the released model and the parameters found in ``eval_fire.yml``, we obtain 90.3 on the validation set, 82.6 and 62.2 on ROxford medium and hard respectively, 85.2 and 70.0 on RParis medium and hard respectively.


## Training a FIRe model 

Simply run
```
python train.py train_fire.yml -e train_fire
```
All training outputs will be saved to ``fire_experiments/train_fire``.

To evaluate the trained model that was saved in ``fire_experiments/train_fire``, simply run:
```
python evaluate.py eval_fire.yml -e train_fire -ml train_fire
```

## Pretrained models

For reproducibility, we provide the following model weights for the architecture we use in the paper (ResNet50 without the last block + LIT):
* Model pre-trained on ImageNet-1K (with Cross-Entropy, the pre-trained model we use for training FIRe) [(link)](http://download.europe.naverlabs.com/ComputerVision/FIRe/pretraining/fire_imagenet.pth)
* Model trained on SfM-120k trained with FIRe [(link)](http://download.europe.naverlabs.com/ComputerVision/FIRe/official/fire.pth)

They will be automatically downloaded when running the training / testing script.

## Dockerfile

For convenience, we provide a dockerfile. You can build it with
```bash
docker build --tag naver/fire .
```
It does not contain the ``fire_data`` nor ``fire_experiments`` so these need to be stored outside.  
In ``evaluate.py``, the options ``--data-folder`` and ``--exp-folder`` can be used to overwrite these paths.

example:
```bash
docker run --gpus all --rm -it --ipc=host --mount type=bind,source=/local/fire,target=/local/fire --entrypoint bash naver/fire
python evaluate.py eval_fire.yml --data-folder /local/fire/fire_data --exp-folder /local/fire/fire_experiments
```

## kapture integration

With ``kapture_compute_pairs.py`` you can compute pairs from datasets that are provided in kapture format ([link to kapture github](https://github.com/naver/kapture)) using FIRe or HOW. These pairs can be used to, e.g., run the kapture visual localization pipeline ([link to kapture-localization github](https://github.com/naver/kapture-localization)).
``--codebook-cache-path`` can be used to cache the codebook. It only needs to be computed once per model.  
``--ivf-cache-path`` can be used to cache the ivf database. It needs to be computed once per model per dataset (mapping images).  
``--model-load``, ``--data-folder`` can be used to overwrite ``demo_eval.net_path`` and ``demo_eval.fire_data``. Note that ``demo_eval.exp_folder`` and ``evaluation.local_descriptor.datasets`` are ignored.

example: extracting top50 FIRe pairs, and top50 HOW pairs for GangnamStation_B2
```
docker run --gpus all --rm -it --ipc=host --mount type=bind,source=/local/fire,target=/local/fire --entrypoint bash naver/fire

# prepare dataset
mkdir /local/fire/kapture_datasets
cd /local/fire/kapture_datasets
kapture_download_dataset.py update
kapture_download_dataset.py install  "GangnamStation_B2*"
# read license terms and type y [enter] to agree
cd GangnamStation/B2/release
kapture_merge.py -v info \
                 -i test validation \
                 -o query_all \
                 --image_transfer link_relative

# extract FIRe pairs
cd /opt/src/fire
# map -> map pairs
python3 kapture_compute_pairs.py -v debug \
    --parameters eval_fire.yml \
    --model fire \
    --data-folder /local/fire/fire_data \
    --codebook-cache-path /local/fire/fire_codebook \
    --ivf-cache-path /local/fire/kapture_datasets/GangnamStation/B2/release/fire_ivf \
    --map /local/fire/kapture_datasets/GangnamStation/B2/release/mapping/ \
    -o /local/fire/kapture_datasets/GangnamStation/B2/release/pairsfile/mapping/fire_top50.txt \
    --topk 50
# query -> map pairs
python3 kapture_compute_pairs.py -v debug \
    --parameters eval_fire.yml \
    --model fire \
    --data-folder /local/fire/fire_data \
    --codebook-cache-path /local/fire/fire_codebook \
    --ivf-cache-path /local/fire/kapture_datasets/GangnamStation/B2/release/fire_ivf \
    --map /local/fire/kapture_datasets/GangnamStation/B2/release/mapping/ \
    --query /local/fire/kapture_datasets/GangnamStation/B2/release/query_all \
    -o /local/fire/kapture_datasets/GangnamStation/B2/release/pairsfile/query/fire_top50.txt \
    --topk 50

# extract HOW pairs
cd /opt/src/fire
# you can use the same data-folder as for fire
# map -> map pairs
python3 kapture_compute_pairs.py -v debug \
    --parameters ../how/examples/params/eccv20/eval_how_r50-_1000.yml \
    --model how \
    --data-folder /local/fire/fire_data \
    --codebook-cache-path /local/fire/how_codebook \
    --ivf-cache-path /local/fire/kapture_datasets/GangnamStation/B2/release/how_ivf \
    --map /local/fire/kapture_datasets/GangnamStation/B2/release/mapping/ \
    -o /local/fire/kapture_datasets/GangnamStation/B2/release/pairsfile/mapping/how_top50.txt \
    --topk 50
# query -> map pairs
python3 kapture_compute_pairs.py -v debug \
    --parameters ../how/examples/params/eccv20/eval_how_r50-_1000.yml --model fire \
    --data-folder /local/fire/fire_data \
    --codebook-cache-path /local/fire/how_codebook \
    --ivf-cache-path /local/fire/kapture_datasets/GangnamStation/B2/release/how_ivf \
    --map /local/fire/kapture_datasets/GangnamStation/B2/release/mapping/ \
    --query /local/fire/kapture_datasets/GangnamStation/B2/release/query_all \
    -o /local/fire/kapture_datasets/GangnamStation/B2/release/pairsfile/query/how_top50.txt \
    --topk 50
```