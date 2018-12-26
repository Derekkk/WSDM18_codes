# Modelling Domain Relationships for Transfer Learning on Retrieval-based Question Answering Systems in E-commerce

(Here we use SNLI+MNLI data, since the AliExpress Data cannot be released.)

This repository contains the following components:

- [SNLI Data] at `put the SNLI data under the fold ../data` (refer to the line 5-14 in preprocess.py)
- [MNLI Data] at `put the MultiNLI data under the fold ../data`
- [hCNN] at `train_transfer_model3.py` & `train_transfer_model5.py` with hCNN as the base model
- [DRSS] at `train_transfer_model3.py`
- [DRSS-Adv] at `train_transfer_model5.py`


### Steps to run the codes:

- [DRSS]
To run this model, you need to set the glove word embedding path to your local path in line 103, and run:
```bash
python train_transfer_model3.py
```

- [DRSS-Adv]
To run this model, you need to set the glove word embedding path to your local path in line 103, and run:
```bash
python train_transfer_model5.py
```


### Requirements

- Python 2.x
- [TensorFlow](https://www.tensorflow.org)
- [Scikit-Learn](http://scikit-learn.org/stable/index.html)
- [Numpy](http://www.numpy.org/)


# License:

Singapore Management University
