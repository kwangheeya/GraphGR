# Envrionment
Windows 10
Python 3.9.13
PyTorch 1.11.0
CUDA 11.3


### Requirements
```
numpy==
pandas==
scikit-learn==
scipy==
torch==
```

### Running the Model
To train and evaluate the model on data <i>e.g.,</i> `gowalla`, please run
```
python train_ConvGR.py --dataset gwl --cuda 
```