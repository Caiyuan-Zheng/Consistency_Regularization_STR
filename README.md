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
sh run_baseline.sh
```

### Testing
```bash
sh run_test.sh
```

### Pretrained Models
|Model|Labeled data|Unlabeled data|Avg score|
|TRBA|Synth90K+SynthText|None|91.38
|TRBA|Synth90K+SynthText|10.6M unlabeled data|94.34
### Recognize a image
```bash
python  pre_img.py  YOUR/MODEL/PATH  YOUR/IMAGE/PATH
```

### Citation
```
@article{yang2020holistic,
  title={A Holistic Representation Guided Attention Network for Scene Text Recognition},
  author={Yang, Lu and Wang, Peng and Li, Hui and Li, Zhen and Zhang, Yanning},
  journal={Neurocomputing},
  year={2020},
  publisher={Elsevier}
}
```
### Acknowledgment
This code is based on [STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels) by [ku21fan]([https://github.com/Canjie-Luo](https://github.com/ku21fan)). Thanks for your contribution.
