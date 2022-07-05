# Consistency_Regularization_STR

It's the code for the paper [Pushing the Performance Limit of Scene Text Recognizer without Human Annotation](https://arxiv.org/abs/2204.07714), CVPR 2022.
Test in Python3.7.
### Install the enviroment
```bash
    pip install -r requirements.txt
```
### Data Prepare
Please convert your own dataset to **LMDB** format by create_dataset.py. (Borrowed from https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py, provided by [Baoguang Shi](https://github.com/bgshih))

For labeled dataset, there are converted [Synth90K](http://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) LMDB dataset by [luyang-NWPU](https://github.com/luyang-NWPU): [[Here]](https://pan.baidu.com/s/1C42j5EoDy1fTtDE8gwwndw),  password: tw3x

For unlabeled dataset, you could download raw images from [imagenet](https://image-net.org/), [places2](http://places2.csail.mit.edu/) and [openimages](https://storage.googleapis.com/openimages), then detect and crop word images
from these images. 

### Supervised Training
```bash
sh run_baseline.sh
```
### Semi-Supervised Training
```bash
sh run_semi.sh
```

### Testing
```bash
sh run_test.sh
```

### Pretrained Models and Results
download pretrained model from here(https://pan.baidu.com/s/1JF97VY0oiPiK5GpDEsYioQ?pwd=abhr, password:abhr) and put them in saved_models.
|Model|Labeled data|Unlabeled data                |IC13 857|IC13 1015|SVT     |IIIT    |IC15 1811|IC15 2077|SVTP   |CUTE    |
| :----: | :----: | :----:                        | :----: | :----:  | :----: | :----: | :----:  | :----: | :----: | :----: |
|TRBA_pr|10% (Synth90K+SynthText)  |-          | 96.3       | 94.3  | 91.5      | 94.3    | 81.5    |77.7 |84.2 |87.5|
|TRBA_pr|10% (Synth90K+SynthText)  |1.06M unlabeled real data     | 97.2    |95.9 | 94.3        |   96.6  |  86.8 |79.7 |89.0 |93.4 |
|TRBA_pr|Synth90K + SynthText|-                     |97.2   | 95.9   | 91.8    |95.5      | 83.6   |  79.7    | 87.3   |88.5|
|TRBA_cr|Synth90K + SynthText|10.6M unlabeled real data|98.0| 96.4   | 96.0   | 97.0   |  88.8   | 84.9   | 90.9  | 95.1   |


### Acknowledgment
This code is based on [STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels) by [Jeonghun Baek](https://github.com/ku21fan). Thanks for your contribution.
