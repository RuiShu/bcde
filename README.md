# bcde
The Bottleneck Conditional Density Estimator provides a semi-supervised learning framework for high-dimensional conditional density estimation. This repository provides code to run experiments in from the 2017 ICML paper [Bottleneck Conditional Density Estimation](https://arxiv.org/abs/1611.08568).

This work was done while interning at Adobe Systems.

## Dependencies

Please make sure to pip install the following dependencies
```
tensorflow-gpu==1.1.0
tensorbayes==0.1.1
```

## Example
To run the model (2-layer hybrid+factored BCDE), simply do:
```
python main.py --model hybrid_factored
```
