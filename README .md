# 2D Human Pose Estimation (MobileNetV3 + PyTorch)

Lightweight deep learning model that predicts the **17 body keypoints** (COCO format) of a single person from a cropped image.

<!-- Add your images to an `images/` folder, then uncomment:
![Loss curve](images/loss_curve.png)
![Sample predictions](images/sample_predictions.png)
-->

## Overview

| | |
|---|---|
| **Task** | 2D keypoint localization (17 joints, x/y coordinates + visibility) |
| **Model** | Pretrained MobileNetV3-Small backbone + convolutional head + two regression heads |
| **Framework** | PyTorch / torchvision |
| **Dataset** | [InfiniteForm](https://runsdata.medium.com/infiniteform-a-synthetic-minimal-bias-dataset-for-fitness-applications-346e69f35b81), an open-source synthetic human pose dataset with COCO-style annotations |
| **Input size** | 224 × 224 RGB crops of one person |
| **Training** | Google Colab (GPU) |

## Pipeline

### 1. Data preparation (`prepare_kit.py`)

The dataset contains full scenes, so each person is cropped before training:

1. Read the COCO-style annotation file and find every non-crowd person in each `*rgb.png` image.
2. Crop the person's bounding box with a 5 px margin.
3. Paste the crop onto a black square canvas (keeps the aspect ratio) and resize it to 224 × 224.
4. Shift and rescale the keypoint coordinates to match the new crop.
5. Save the image to `crop_pic/` and its labels (`x_y` coordinates and `confidence`/visibility flags) to `crop_label/` as JSON.

### 2. Model (`PoseEstimationWithBackbone`)

- **Backbone:** MobileNetV3-Small pretrained on ImageNet
- **Neck:** two 3×3 conv blocks (576 → 256 → 128 channels) with BatchNorm and Hardswish
- **Head 1:** fully connected layers predicting 17 × 2 normalized coordinates
- **Head 2:** fully connected layers predicting a visibility score for each of the 17 keypoints

### 3. Training

- **Loss:** L1 loss on the coordinates of labeled keypoints + Smooth L1 loss on the visibility scores
- **Optimizer:** AdamW (lr = 1e-4, weight decay = 1e-4), batch size 256
- **Scheduler:** MultiStepLR (halves the learning rate at epochs 2, 10, 20, 50, 70, 80, 90, 95)
- **Augmentation:** random choice between Gaussian blur and random erasing, plus ImageNet normalization
- **Split:** 80% train / 20% test (`random_state=42`)
- **Epochs:** 100

## Results

Evaluated on the held-out 20% test split:

| Metric | Value |
|---|---|
| Train loss (epoch 100) | ≈ 0.0531 |
| Test loss | ≈ 0.0571 |
| Keypoint accuracy (PCK@0.5) | **≈ 81.3%** |

**Accuracy definition:** a keypoint counts as correct if the predicted point is within 0.5 × the shoulder distance of the ground truth (hip distance if the shoulders are not labeled) and the model predicts it as visible. Keypoints without a label are excluded.

## Repository structure

```
.
├── README.md
├── prepare_kit.py            # crops people from full images and rescales labels
├── pose_estimation.ipynb     # dataset class, model, training, evaluation
├── requirements.txt
└── .gitignore
```

## How to run

1. Download the [InfiniteForm](https://runsdata.medium.com/infiniteform-a-synthetic-minimal-bias-dataset-for-fitness-applications-346e69f35b81) dataset.
2. Create the output folders, then crop the images:

```python
import os
from prepare_kit import cut_bbox

images_dir = "path/to/infiniteform/images"
annotations = "path/to/annotations.json"
out_dir = "data"

os.makedirs(os.path.join(out_dir, "crop_pic"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "crop_label"), exist_ok=True)

cut_bbox(os.listdir(images_dir), images_dir, annotations, out_dir, 224, 224)
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Open `pose_estimation.ipynb` and set `DATA_PATH` to your `data` folder (the notebook was written for Google Colab, so adjust the Google Drive paths if you run it locally).

## Pretrained weights

Final checkpoint (epoch 100): [Google Drive](https://drive.google.com/file/d/10TysYr0w6PjrvxJanNWK4C1eAvDjYBBA/view?usp=sharing)

## Tools

Python, PyTorch, torchvision, NumPy, Pillow, scikit-learn, Matplotlib, tqdm

## Author

**Farah Youssef** · [LinkedIn](https://linkedin.com/in/farah-yousef-66031a328)
