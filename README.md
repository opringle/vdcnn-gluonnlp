# Gluon Deep Char CNN

MXNet Gluon code to both train and optimize a deep convolutional neural network for the task of text classification.

The model achieves XX% test accuracy on the AG News dataset (state of the art is XX%)

# Instructions

- Create a virtual env
    - `mkvirtualenv -a ./ -r requirements.txt -p python3.6 gluon`
- To train the model with best parameters found through bayesian optimization
    - `MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python train.py --train data/ag_news --val data/ag_news --gpus=1`
- To run an Amazon Sagemaker bayesian optimization job
    - Setup sagemaker
    -  `python sage.py <jobname> <aws profile>`

# ToDo

- [x] Test GPU training
- [x] Add logging statements
- [x] Argparse in groups
- [x] Evaluation function
- [x] Change to VDCNN
- [x] Document functions
- [x] Sagemaker hyperopt
- [x] Bucketing
- [ ] Batch size causes hyperopt job to die
- [ ] Loss bug when training on ag_news
- [ ] Custom kmax pool layer
- [ ] Label smoothing
- [ ] Sagemaker deployment
- [ ] [Multi GPU training](https://medium.com/apache-mxnet/94-accuracy-on-cifar-10-in-10-minutes-with-amazon-sagemaker-754e441d01d7)
