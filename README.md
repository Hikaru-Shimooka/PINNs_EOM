# PINNs
このコードは、初期条件を教師データの損失関数、運動方程式を方程式の損失関数にするPINNsです。  
This code represents Physics-Informed Neural Networks (PINNs), where the initial conditions are used as the loss function for the teacher data and the governing equations are used as the loss function for the equations.

## 環境 (Environment)
このコードは、以下の環境で動作確認済みです。  
This code has been tested and verified to work with the following versions.
- [Python 3.10.5](https://www.python.org/downloads/release/python-3105/)  
- [PyTorch 1.12.1](https://pytorch.org/get-started/previous-versions/)

## 利用方法 (Usage)
以下のコマンドを実行する。
```
$ python train.py
```

## 参考文献 (Reference)
- PINNs [M. Raissi, P. Perdikaris, G.E. Karniadakis, Journal of Computational Physics, Volume 378, 2019](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)
- コード <https://github.com/kimy-de/HFM>