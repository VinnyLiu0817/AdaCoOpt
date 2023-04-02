## AdaCoOpt

This repository includes source code for the paper 'AdaCoOpt: Leverage the Interplay of Batch Size and Aggregation Frequency for Federated Learning'. 
The code was mainly re-written based on a distinguished and excellent source code from S. Wang, T. Tuor, T. Salonidis, K. K. Leung, C. Makaya, T. He, and K. Chan, "Adaptive federated learning in resource constrained edge computing systems," IEEE Journal on Selected Areas in Communications, vol. 37, no. 6, pp. 1205 – 1221, Jun. 2019. 

### Prerequisites

The code runs on Python 3 with Tensorflow version 1 (>=1.13). To install the dependencies, run
```
pip3 install -r requirements.txt
```

Then, download the datasets manually and put them into the `datasets` folder.
- For MNIST dataset, download from <http://yann.lecun.com/exdb/mnist/> and put the standalone files into `datasets/mnist`.
- For CIFAR-10 dataset, download from <https://www.cs.toronto.edu/~kriz/cifar.html>, extract the standalone `*.bin` files and put them into `datasets/cifar-10-batches-bin`.

To test the code: 
- Run `server.py` and wait until you see `Waiting for incoming connections...` in the console output.
- Run 3 parallel instances of `client2.py` on the same machine as the server. 
- You will see console outputs on both the server and clients indicating message exchanges. The code will run for a few minutes before finishing.

### Code Structure

All configuration options are given in `config.py` which also explains the different setups that the code can run with.

The results are saved in the `results` folder. 

Currently, the supported datasets are MNIST and CIFAR-10, and the supported models are SVM and CNN. The code can be extended to support other datasets and models too.  

### Citation

When using this code for scientific publications, please kindly cite the above paper.

### Contributors

This code was written by Weijie Liu based on a distingushed work of Shiqiang Wang and Tiffany Tuor.
