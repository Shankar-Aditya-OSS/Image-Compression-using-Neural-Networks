## Residual RNN

This repo is a Tensorflow implementation of paper [Full Resolution Image Compression with Recurrent Neural Networks](https://arxiv.org/pdf/1608.05148.pdf). This repo also contains a trained Residual LSTM model (saved in `save/`) on a small dataset.

Original image:

![original](Lena.png)

Reconstructed:

![reconstruction](Compressed.png)

### Requirements
- SciPy - 1.4.1
- Tensorflow 2.5.0

Before executing make sure to change folder locations in the code accordingly.
### Usage(without GUI)
Training, put training data in `Dataset` folder and run following code for default setting. The model parameter can be modified in `model.py`. Running time comparison of original and reconstructed images can be seen in `eval/`. Model file is saved in `save/model`.

```
python train.py -f imgs
```

Encoding
```
python encode.py --model save/model --input Lena.png --iters 10 --output compressed.npz
```

Decoding
```
python decode.py --model save/model --input compressed.npz --output compressed.png
```

Evaluation, code from Tensorflow's official [repo](https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py)

```
python msssim.py -o Lena.png -c Compressed.png
```

#Usage (with GUI):
```
1.Open cmd and head to project folder.
2.Type:
```
python GUI.py
```
3.A tkinter window will pop up.
4.Browse to Dataset folder and select an image, only png images are accepted.
5.Click on Encode button, a binary file will be created and saved in the folder
6.Click on Decode button, the binary file will be accessed and the image will be re-constructed from that data.
7.By clicking on Eval button, it will compare the original image and re-constructed image,
8.MSSIM: 0.96 - 1 are said to be the good values,which means the re-constructed image is almost similar to the original image.
```
# Image-Compression-using-Neural-Networks
