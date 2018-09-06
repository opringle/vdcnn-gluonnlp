# Gluon Deep Char CNN

MXNet GluonNlp code to both train and optimize a deep convolutional neural network for the task of text classification.

The model achieves XX% test accuracy on the AG News dataset (state of the art is XX%)

# Instructions

- Create a virtual env
    - `mkvirtualenv -a ./ -r requirements.txt -p python3.6 gluon`
- To train the model on a gpu with best parameters found through bayesian optimization
    - `MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python vdcnn/train.py --train data/ag_news --val data/ag_news --gpus=1`
- To run the Amazon Sagemaker bayesian optimization job
    - [Setup sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html)
    - Build the custom docker image: `bash build_and_push.sh vdcnn`
    - Start your sagemaker hyperopt job: `python sage.py <jobname> <aws profile> <image_repo>`

# ToDo

- [x] Test GPU training
- [x] Add logging statements
- [x] Argparse in groups
- [x] Evaluation function
- [x] Change to VDCNN
- [x] Document functions
- [x] Bucketing
- [ ] Create and reference custom docker image
- [ ] Sagemaker hyperopt get metric to record
- [ ] Batch size causes hyperopt job to die
- [ ] Script to upload data to s3 from repo
- [ ] Sagemaker deployment
- [ ] [Multi GPU training](https://medium.com/apache-mxnet/94-accuracy-on-cifar-10-in-10-minutes-with-amazon-sagemaker-754e441d01d7)
