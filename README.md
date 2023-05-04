<!-- # DMRIB
The official repos for ""Disentangling Multi-view Representations Beyond Inductive Bias"" (DMRIB) -->


## Training step

We show that how `DMRIB` train on the `EdgeMnist` dataset.

Before the training step, you need to set the `CUDA_VISIBLE_DEVICES`, because of the `faiss` will use all gpu. It means that it will cause some error if you using `tensor.to()` to set a specific device.

1. set environment.
```
export CUDA_VISIBLE_DEVICES=0
```

2. train the pretext model.
First, we need to run the pretext training script `src/train_pretext.py`. We use simclr-style to training a self-supervised learning model to mine neighbors information. The pretext config commonly put at `configs/pretext`. You just need to run the following command in you terminal:
```
python train_pretext.py -f ./configs/pretext/pretext_EdgeMnist.yaml
```

3. train the self-label clustering model.
Then, we could use the pretext model to training clustering model via `src/train_scan.py`.
```
python train_scan.py -f ./configs/scan/scan_EdgeMnist.yaml
```
After that, we use the fine-tune script to train clustering model `scr/train_selflabel.py`.
```
python train_selflabel.py -f ./configs/scan/selflabel_EdgeMnist.yaml
```

4. training the view-specific encoder and disentangled.
Finally, we could set the self-label clustering model as the consisten encoder. And train the second stage via `src/train_dmrib.py`.
```
python train_dmrib.py -f ./configs/dmrib/dmrib_EdgeMnist.yaml
```


## Validation

Note: you can find the pre-train weigths at [here](https://drive.google.com/file/d/1Q8u9_SlAgebI03guE0hfkxgrmBE5sy8p/view?usp=sharing).

```
python validate.py -f ./configs/dmrib/dmrib_EdgeMnist.yaml
```


## Credit

Thanks:
```Van Gansbeke, Wouter, et al. "Scan: Learning to classify images without labels." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part X. Cham: Springer International Publishing, 2020.```