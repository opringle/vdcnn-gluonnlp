import sagemaker
from sagemaker.mxnet import MXNet
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
import argparse
import boto3

parser = argparse.ArgumentParser(description="MXNet + Sagemaker hyperparameter optimization job",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--job-name', type=str, required=True,
                    help='name of job')
parser.add_argument('--profile', type=str, default='',
                    help='role with permissions to run sagemaker')
parser.add_argument('--role_arn', type=str,
                    default='arn:aws:iam::430515702528:role/service-role/AmazonSageMaker-ExecutionRole-20180730T100605',
                    help='arn of sagemaker execution role')
parser.add_argument('--bucket_name', type=str, default='finn-dl-sandbox-atlas',
                    help='bucket to store code, data and artifacts')
parser.add_argument('--train-code', type=str, default='./train.py',
                    help='python module containing train() function')
parser.add_argument('--source-dir', type=str, default='./',
                    help='directory of other python modules imported')
parser.add_argument('--train-instance-type', type=str, default='local',
                    help='instance type for training')
parser.add_argument('--train-instance-count', type=int, default=1,
                    help='number of instances to distribute training')
parser.add_argument('--max-jobs', type=int, default=500,
                    help='number of hypesrparameter jobs to run')
parser.add_argument('--max-parallel-jobs', type=int, default=1,
                    help='number of parallel jobs to run')
parser.add_argument('--data-dir', type=str, default='./data/ag_news',
                    help='path to train/test pickle files')


if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Assume permissions to provision services with Sagemaker role
    session = boto3.Session(profile_name='sagemaker_execution')
    sagemaker_session = sagemaker.Session(boto_session=session)
    local_session = sagemaker.local.local_session.LocalSession(boto_session=session)

    # Initialize variables
    custom_code_upload_location = 's3://{}/customcode/mxnet'.format(args.bucket_name)
    model_artifacts_location = 's3://{}/artifacts'.format(args.bucket_name)
    train_path = 's3://{}/{}/{}'.format(args.bucket_name, args.data_dir, 'train.pickle')
    val_path = 's3://{}/{}/{}'.format(args.bucket_name, args.data_dir, 'test.pickle')

    # Hyperparameters to search
    search_space = {'learning_rate': ContinuousParameter(0.0001, 0.5),
                    'momentum': ContinuousParameter(0.8, 0.999),
                    'batch_size': IntegerParameter(8, 512),
                    'dropout': ContinuousParameter(0.1, 0.9),
                    'smooth_alpha': ContinuousParameter(0.0, 0.1)
                    }

    # Hyperparameters to fix
    hyperparameters = {'epochs': 10
                       }

    # Create an estimator
    estimator = MXNet(sagemaker_session=sagemaker_session if 'local' not in args.train_instance_type else local_session,
                      hyperparameters=hyperparameters,
                      entry_point=args.train_code,
                      source_dir=args.source_dir,
                      role=args.role_arn,
                      output_path=model_artifacts_location,
                      code_location=custom_code_upload_location,
                      train_instance_count=args.train_instance_count,
                      train_instance_type=args.train_instance_type,
                      base_job_name=args.job_name,
                      py_version='py3',
                      framework_version='1.1.0',
                      train_volume_size=1)

    # Configure Hyperparameter Tuner
    my_tuner = HyperparameterTuner(estimator=estimator,
                                   objective_metric_name='Validation-accuracy',
                                   hyperparameter_ranges=search_space,
                                   metric_definitions=[
                                       {'Name': 'Validation-accuracy', 'Regex': 'Best Validation Accuracy =(\d\.\d+)'}],
                                   max_jobs=args.max_jobs,
                                   max_parallel_jobs=args.max_parallel_jobs)

    # Start hyperparameter tuning job
    # my_tuner.fit({'train': train_path, 'val': val_path})
    estimator.fit({'train': train_path, 'val': val_path})
