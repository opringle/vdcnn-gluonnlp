# gluon-examples
experimenting with MXNet Gluon API

# ToDo

- [x] Test GPU training
- [x] Add logging statements
- [x] Argparse in groups
- [x] Evaluation function
- [x] Change to VDCNN
- [x] Document functions
- [ ] Sagemaker hyperopt
- [ ] Custom kmax pool layer
- [ ] Label smoothing
- [ ] Add bucketing
- [ ] Sagemaker deployment
- [ ] [Multi GPU training](https://medium.com/apache-mxnet/94-accuracy-on-cifar-10-in-10-minutes-with-amazon-sagemaker-754e441d01d7)

# To run the code

`mkvirtualenv -a ./ -r requirements.txt -p python3.6 gluon`
`python train.py --train ../VDCNN-for-text-classification/data/atb_model_41/train.pickle --val ../VDCNN-for-text-classification/data/atb_model_41/test.pickle`
