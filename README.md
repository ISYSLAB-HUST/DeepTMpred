## Improving the topology prediction of alpha-helical transmembrane proteins with transfer learning

### Abstract
We consider that the pre-training language model can make most use of massive unlabeled protein sequence data to learn 
general feature representations for TMPs. Therefore, we proposed a transfer learning method, DeepTMpred, using pre-trained 
self-supervised language models called ESM, convolutional neural networks, and conditional random fields for alpha-TMP topology prediction. 
Compared with other tools, DeepTMpred can achieve state-of-the-art results and obtain pretty good prediction results for TMPs 
lacking sufficient evolutionary information.

### Dependencies
```shell script
pip install git+https://github.com/facebookresearch/esm.git
pip install -r requirements.txt
```
For reproducibility, we listed the packages used for generating the results in the paper, 
but other versions of these packages will likely give similar results.

### Dataset
Orientations of Proteins in Membranes (OPM) database: https://opm.phar.umich.edu/download

### TMH train and prediction
```shell script
python tmh_main.py &
```

### Orientation train and prediction
```shell script
python orientaion_main.py &
```

### License
[MIT](LICENSE)