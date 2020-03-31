# Original hardcoded implementation of [AutoDot](https://github.com/oxquantum-repo/AutoDot)
### If you want to perform automated tuning it is highly recomended that you use the [AutoDot](https://github.com/oxquantum-repo/AutoDot) which is a very configurable refactored implementation of this repo.

**NOTE: DUE TO MULTIPROCESSING PACKAGE THE CURRENT IMPLEMENTATION ONLY WORKS ON UNIX/LINUX OPERATING SYSTEMS**

The quantum devices used to implement spin qubits in semiconductors can be challenging to tune and characterise. Often the best approaches to tuning such devices is manual tuning or a simple heuristic algorithm which is not flexible across devices. This repository contains the original code used in the statistical tuning approach detailed in https://arxiv.org/abs/2001.02589. This code is highly hardcoded and hence difficult to addapt to your specific problem we recomend using the refactored code [AutoDot](https://github.com/oxquantum-repo/AutoDot) as it's implementation is far more general and documentation is also provided for running it, hence can be addapeted and used easilly.

# Prerequisites
It needs follwing python packages. Use anaconda or pip to install them.
```
numpy
scipy
matplotlib
pyDOE
GPy # when to use GPy
```

# Running the algorithm
The main driver of the algorithm is `test_sampling_gpc.py`.
