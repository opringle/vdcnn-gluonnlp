import sagemaker
from sagemaker.mxnet import MXNet
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
import argparse
import boto3

parser = argparse.ArgumentParser(description="MXNet + Sagemaker hyperparameter optimization job",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Authentication
group = parser.add_argument_group('Authentication args')
group.add_argument('--profile', type=str, default='sagemaker_execution',
                    help='role with permissions to run sagemaker')
group.add_argument('--role_arn', type=str,
                    default='arn:aws:iam::430515702528:role/service-role/AmazonSageMaker-ExecutionRole-20180730T100605',
                    help='arn of sagemaker execution role')

# Data and code
group = parser.add_argument_group('Data and code arguments')
group.add_argument('--bucket_name', type=str, default='finn-dl-sandbox-atlas',
                    help='bucket to store code, data and artifacts')
group.add_argument('--data-dir', type=str, default='atb_model_46/strat_split/data',
                    help='path to train/test pickle files')
group.add_argument('--train-code', type=str, default='train.py',
                    help='python module containing train() function')
group.add_argument('--source-dir', type=str, default='vdcnn',
                    help='directory of other python modules imported')

# Job details
group = parser.add_argument_group('Job arguments')
group.add_argument('job_name', type=str,
                    help='name of job')
group.add_argument('--image-name', type=str,
                    help='name of image repo')
group.add_argument('--train-instance-type', type=str, default='local',
                    help='instance type for training')
group.add_argument('--train-instance-count', type=int, default=1,
                    help='number of instances to distribute training')
group.add_argument('--max-jobs', type=int, default=500,
                    help='number of hyperparameter jobs to run')
group.add_argument('--max-parallel-jobs', type=int, default=1,
                    help='number of parallel jobs to run')


if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Assume permissions to provision services with Sagemaker role
    session = boto3.Session(profile_name=args.profile)
    sagemaker_session = sagemaker.Session(boto_session=session)
    local_session = sagemaker.local.local_session.LocalSession(boto_session=session)

    # Initialize variables
    custom_code_upload_location = 's3://{}/customcode/mxnet'.format(args.bucket_name)
    model_artifacts_location = 's3://{}/artifacts'.format(args.bucket_name)
    data_path = 's3://{}/{}'.format(args.bucket_name, args.data_dir)

    # Hyperparameters to search
    search_space = {'min_lr': ContinuousParameter(0.0001, 1),
                    'max_lr': ContinuousParameter(0.0001, 1),
                    'lr_cycle_epochs': ContinuousParameter(1, 10),
                    'lr_increase_fraction': ContinuousParameter(0.1, 0.5),
                    'momentum': ContinuousParameter(0.8, 0.999),
                    'dropout': ContinuousParameter(0.01, 0.99),
                    'temp_conv_filters': IntegerParameter(32, 128),
                    'l2': ContinuousParameter(0.0, 0.2)
                    }

    # Hyperparameters to fix
    hyperparameters = {'epochs': 30,
                       'batch_size': 24,
                       'num_buckets': 30,
                       'batch_seq_ration': 0.5,
                       'embed_size': 16,
                       'blocks': [1, 1, 1, 1]
                       }

    # Create an estimator
    estimator = MXNet(image_name=args.image_name,
                      sagemaker_session=sagemaker_session if 'local' not in args.train_instance_type else local_session,
                      hyperparameters=hyperparameters,
                      entry_point=args.train_code,
                      source_dir=args.source_dir,
                      role=args.role_arn,
                      output_path=model_artifacts_location,
                      # code_location=custom_code_upload_location,
                      train_instance_count=args.train_instance_count,
                      train_instance_type=args.train_instance_type,
                      base_job_name=args.job_name,
                      train_volume_size=1)

    # Configure Hyperparameter Tuner
    my_tuner = HyperparameterTuner(estimator=estimator,
                                   objective_metric_name='Best Validation Accuracy',
                                   hyperparameter_ranges=search_space,
                                   metric_definitions=[
                                       {'Name': 'Best Validation Accuracy', 'Regex': 'Best Validation Accuracy =(\d\.\d+)'}],
                                   max_jobs=args.max_jobs,
                                   max_parallel_jobs=args.max_parallel_jobs,
                                   base_tuning_job_name=args.job_name)

    # Start hyperparameter tuning job
    # my_tuner.fit({'train': data_path, 'val': data_path})
    estimator.fit({'train': data_path, 'val': data_path})
