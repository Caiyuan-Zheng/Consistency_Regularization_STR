# Consistency_Regularization_STR

It's the code for the paper [Pushing the Performance Limit of Scene Text Recognizer without Human Annotation]([https://arxiv.org/abs/1904.01375](https://arxiv.org/abs/2204.07714)), Neurocomputing 2020.
Test in Python3.7.
### Install the enviroment
```bash
    pip install -r requirements.txt
```
Please convert your own dataset to **LMDB** format by create_dataset.py. (Borrowed from https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py, provided by [Baoguang Shi](https://github.com/bgshih))

There are converted [Synth90K](http://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) LMDB dataset by [Caiyuan Zheng](https://github.com/luyang-NWPU): [[Here]](https://pan.baidu.com/s/1ATcmCbPh6jPMorI3tDkmOg?pwd=adbf),  password: adbf

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
This code is based on [MORAN](https://github.com/Canjie-Luo/MORAN_v2) by [Canjie-Luo](https://github.com/Canjie-Luo). Thanks for your contribution.
