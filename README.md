# SCAF-Net
A Scene Context Attention-Based Fusion Network for Vehicle Detection

## Installation

```
git clone https://github.com/minghuicode/SCAF-Net
cd SCAF-Net
conda env create -f environment.yml 
```

## Dataset Downloads

downloads DLR-3K dataset at [dlr.de](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-12760/22294_read-52777)

```
cd SCAF-Net/data
wget https://pba-freesoftware.eoc.dlr.de/MunichDatasetVehicleDetection-2015-old.zip
unzip MunichDatasetVehicleDetection-2015-old.zip
ln -sf MunichDatasetVehicleDetection-2015-old/Train dlr
```

## Model Training

There are total 10 labeled aerial images. We use 5 of them for training, others for test.

```
cd SCAF-Net
conda activate torch
python train.py
```

## Model Evaluation

To evaluate model performance on other 5 labeled aerial images, just run test file.

```
cd SCAF-Net
conda activate torch
python test.py --evaluation
```

## Predict

To predict several unseen aerial images, run test files as follow.
Visual output will be saved at `output` folder.

```
cd SCAF-Net
conda activate torch
mkdir input
cp data/dlr/*JPG input/
python test.py
``` 
