# Cấu trúc thư mục
<img src="img/dir.png" >

# Link data train
[Data](https://drive.google.com/drive/folders/1WHBKOkSgadoDB0uor4yVGFpE4yktvRdv?usp=sharing)

#Train
```bash
$ cd Data
$ pip install -U git+https://github.com/albu/albumentations --no-cache-dir
$ cd ..
$ pip install segmentation-models-pytorch
$ python train.py --save ./model.pth
$ python test.py --model ./model.pth
$ python detect.py --model ./model.pth
```
[weight](https://drive.google.com/file/d/14IQ7z0l3AWSl__gJ6CGPDXXyn3zBFYGD/view?usp=sharing)
