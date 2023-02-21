import sagemaker
import boto3
from sagemaker.parameter import CategoricalParameter, ContinuousParameter, IntegerParameter, ParameterRange
from sagemaker.tuner import HyperparameterTuner
from sagemaker.serializers import CSVSerializer
from sagemaker.predictor import Predictor

import numpy as np

#function to generate and train a classification xgboost tuning job
def get_xgb_classifier(region, bucket, prefix, sess, role, hyperparameter_ranges):
    from sagemaker.amazon.amazon_estimator import get_image_uri
    s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
    s3_input_validation = sagemaker.TrainingInput(s3_data='s3://{}/{}/validation'.format(bucket, prefix), content_type='csv')

    xgb_cont = get_image_uri(region, 'xgboost', repo_version='1.0-1')

    xgb = sagemaker.estimator.Estimator(xgb_cont, role, train_instance_count=1, train_instance_type='ml.m4.xlarge',
                                        output_path='s3://{}/{}/output'.format(bucket, prefix), sagemaker_session=sess)

    xgb.set_hyperparameters(eval_metric='auc',
                            objective='binary:logistic',
                            num_round=100,
                            rate_drop=0.1,
                            tweedie_variance_power=1.4)

    tuner = HyperparameterTuner(xgb, 
                                objective_metric_name='validation:auc', 
                                objective_type='Maximize',
                                hyperparameter_ranges=hyperparameter_ranges, 
                                max_jobs=15, max_parallel_jobs=3)

    tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})
    
    return tuner

#function to make classification xgboost prediction
def predict_xgb_cls(xgboost_model, data):
    xgboost_model.serializer = CSVSerializer()
    predictions = xgboost_model.predict(data).decode('utf-8') # predict!
    predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
    return np.round(predictions_array)


def get_ll_classifier(region, bucket, prefix, sess, role, hyperparameter_ranges):
    from sagemaker.amazon.amazon_estimator import get_image_uri
    s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='text/csv')
    s3_input_validation = sagemaker.TrainingInput(s3_data='s3://{}/{}/validation'.format(bucket, prefix), content_type='text/csv')

    ll_cont = get_image_uri(region, 'linear-learner', repo_version='1.0-1')

    ll = sagemaker.estimator.Estimator(ll_cont, role, train_instance_count=1, train_instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix), sagemaker_session=sess)

    ll.set_hyperparameters(predictor_type='binary_classifier')

    tuner = HyperparameterTuner(ll, 
                            objective_metric_name='validation:binary_f_beta', 
                            objective_type='Maximize',
                            hyperparameter_ranges=hyperparameter_ranges, 
                            max_jobs=15, max_parallel_jobs=3)

    tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})
    
    return tuner

def predict_ll_cls(ll_model, data):
    import json
    ll_model.serializer = CSVSerializer()
    predictions = ll_model.predict(data).decode('utf-8') # predict!
    ll_test_pred = json.loads(predictions)
    ll_test_pred = (np.array([yh['score'] for yh in ll_test_pred['predictions']]) > 0.5) * 1
    return ll_test_pred

#function to generate and train a regression xgboost tuning job
def get_xgb_regressor(region, bucket, prefix, sess, role, hyperparameter_ranges):
    from sagemaker.amazon.amazon_estimator import get_image_uri
    s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
    s3_input_validation = sagemaker.TrainingInput(s3_data='s3://{}/{}/validation'.format(bucket, prefix), content_type='csv')

    xgb_cont = get_image_uri(region, 'xgboost', repo_version='1.0-1')

    xgb = sagemaker.estimator.Estimator(xgb_cont, role, train_instance_count=1, train_instance_type='ml.m4.xlarge',
                                        output_path='s3://{}/{}/output'.format(bucket, prefix), sagemaker_session=sess)

    xgb.set_hyperparameters(eval_metric='rmse',
                            objective='reg:squarederror',
                            num_round=100,
                            rate_drop=0.1,
                            tweedie_variance_power=1.4)

    tuner = HyperparameterTuner(xgb, 
                                objective_metric_name='validation:rmse', 
                                objective_type='Minimize',
                                hyperparameter_ranges=hyperparameter_ranges, 
                                max_jobs=15, max_parallel_jobs=3)

    tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})
    
    return tuner

#function to make classification xgboost prediction
def predict_xgb_reg(xgboost_model, data):
    xgboost_model.serializer = CSVSerializer()
    predictions = xgboost_model.predict(data).decode('utf-8') # predict!
    predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
    return predictions_array

def get_ll_regressor(region, bucket, prefix, sess, role, hyperparameter_ranges):
    from sagemaker.amazon.amazon_estimator import get_image_uri
    s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='text/csv')
    s3_input_validation = sagemaker.TrainingInput(s3_data='s3://{}/{}/validation'.format(bucket, prefix), content_type='text/csv')

    ll_cont = get_image_uri(region, 'linear-learner', repo_version='1.0-1')

    ll = sagemaker.estimator.Estimator(ll_cont, role, train_instance_count=1, train_instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix), sagemaker_session=sess)

    ll.set_hyperparameters(predictor_type='regressor')

    tuner = HyperparameterTuner(ll, 
                            objective_metric_name='validation:objective_loss', 
                            objective_type='Minimize',
                            hyperparameter_ranges=hyperparameter_ranges, 
                            max_jobs=15, max_parallel_jobs=3)

    tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})
    
    return tuner

def predict_ll_reg(ll_model, data):
    import json
    ll_model.serializer = CSVSerializer()
    predictions = ll_model.predict(data).decode('utf-8') # predict!
    ll_test_pred = json.loads(predictions)
    ll_test_pred = np.array([yh['score'] for yh in ll_test_pred['predictions']])
    return ll_test_pred
