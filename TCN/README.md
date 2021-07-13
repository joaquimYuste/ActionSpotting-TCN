# Temporal Convolutional Networks with Attention Module
This repository uses code of [Alleviating Over-segmentation Errors by Detecting Action Boundaries](https://github.com/yiskw713/asrf).

## Dataset

GTEA, 50Salads, Breakfast  

You can download features and G.T. of these datasets from [this repository](https://github.com/yabufarha/ms-tcn).  

## Requirements

* Python = 3.7
* pytorch => 1.0
* torchvision
* pandas
* numpy
* Pillow
* PyYAML

You can download packages using requirements.txt.  

```bash
pip install -r requirements.txt
```

## Directory Structure

```directory structure
root ── csv/
      ├─ libs/
      ├─ imgs/
      ├─ result/
      ├─ utils/
      ├─ dataset ─── 50salads/...
      │           ├─ breakfast/...
      │           └─ gtea ─── features/
      │                    ├─ groundTruth/
      │                    ├─ splits/
      │                    └─ mapping.txt
      ├.gitignore
      ├ README.md
      ├ requirements.txt
      ├ save_pred.py
      ├ train.py
      └ evaluate.py
```

1. You can train and evaluate models specifying a configuration file generated in the above process like:

    ```bash
    python train.py ./result/50salads/dataset-50salads_split-1/config.yaml
    python evaluate.py ./result/50salads/dataset-50salads_split-1/config.yaml --refinement_method refinement_with_boundary
    ```

1. You can also save model predictions as numpy array by running:

    ```bash
    python save_pred.py ./result/50salads/dataset-50salads_split-1/config.yaml --refinement_method refinement_with_boundary
    ```

1. If you want to visualize the saved model predictions, please run:

    ```bash
    python utils/convert_arr2img.py ./result/50salads/dataset-50salads_split1/predictions
    ```


## Reference
* Yuchi Ishikawa, Seito Kasai, Yoshimitsu Aoki, Hirokatsu Kataoka, "Alleviating Over-segmentation Errors by Detecting Action Boundaries" in WACV 2021 ([paper](https://arxiv.org/pdf/2007.06866.pdf))
* Colin Lea et al., "Temporal Convolutional Networks for Action Segmentation and Detection", in CVPR2017 ([paper](http://zpascal.net/cvpr2017/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.pdf))
* Yazan Abu Farha et al., "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation", in CVPR2019 ([paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.pdf), [code](https://github.com/yabufarha/ms-tcn))
