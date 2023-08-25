# DMRIB
The official repos for ""Disentangling Multi-view Representations Beyond Inductive Bias"" (DMRIB)

- Status: Accepted in ACM MM 2023.


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

Note: you can find the pre-train weigths at [here](https://drive.google.com/file/d/1Q8u9_SlAgebI03guE0hfkxgrmBE5sy8p/view?usp=sharing). And put the pretrained models into the following folders `path to/{config.train.log_dir}/{results}/{config.dataset.name}/eid-{config.experiment_id}/dmrib/final_model.pth`, respectively. For example, if you try to validate the `EdgeMnist` dataset, the default folder is `./experiments/results/EdgeMnist/eid-0/dmrib`. And then, put the pretrained model `edge-mnist.pth` into this folder and rename it to `final_model.pth`. 

If you do not want to use the default setting, you have to modify the line 58 of  the `validate.py`.

```
python validate.py -f ./configs/dmrib/dmrib_EdgeMnist.yaml
```


## Credit

Thanks:
```Van Gansbeke, Wouter, et al. "Scan: Learning to classify images without labels." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part X. Cham: Springer International Publishing, 2020.```

## Citation

```
Guanzhou Ke, Yang Yu, Guoqing Chao, Xiaoli Wang, Chenyang Xu,
and Shengfeng He. 2023. Disentangling Multi-view Representations Be-
yond Inductive Bias. In Proceedings of the 31st ACM International Conference
on Multimedia (MM ’23), October 29–November 3, 2023, Ottawa, ON, Canada.
ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3581783.3611794
```
