import sklearn
import lightgbm
# from sklearn.externals import joblib
import joblib
import pandas as pd
import collections
import math
import warnings
import time
import os
import json
from datetime import datetime

class PredictiveModel:
    """
    Predictive model class is a wrapper for scikit learn regression models
    ref: http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
    """

    def __init__(self, algorithm, sensor, prediction_horizon, evaluation_periode, error_metrics, split_point,
                 retrain_period = None, samples_for_retrain = None, retrain_file_location = None):
        self.algorithm = algorithm
        self.model = eval(self.algorithm)
        self.sensor = sensor
        self.horizon = prediction_horizon
        self.eval_periode = evaluation_periode
        self.split_point = split_point
        self.err_metrics = error_metrics
        self.measurements = collections.deque(maxlen=self.eval_periode)
        self.predictions = collections.deque(maxlen=(self.eval_periode + self.horizon))
        self.predictability = None

        # Retrain configurations
        self.samples_for_retrain = samples_for_retrain
        self.retrain_period = retrain_period
        self.samples_from_retrain = 0
        self.samples_in_train_file = 0
        self.retrain_memory = {"timestamp": [], "ftr_vector": []}


        if(self.retrain_period is not None):
            # Initialize file
            filename = "{}_{}h_retrain.json".format(sensor, prediction_horizon)
            self.train_file_path = os.path.join(retrain_file_location, filename)
            open(self.train_file_path, "w").close()

    def fit(self, filename):

        with open(filename) as data_file:
            #data = pd.read_json(data_file)
            data = pd.read_json(data_file, lines=True) # if not valid json
            # set datetime as index
            data.set_index('timestamp',inplace=True)
            
            # transform ftr_vector from array to seperate fields
            data = data['ftr_vector'].apply(pd.Series)
            
            #print(data)
            # get features
            all_features = list(data)

            # prepare target based on prediction horizon (first one is measurement to shift)
            measurements = data[[data.columns[0]]]
            # this line removed duplicate target values; makes no sense ...
            # removed by Klemen Kenda, 2020/09/09
            #measurements = measurements.loc[~measurements.duplicated(keep='first')]
            data['target'] = measurements.shift(periods = -self.horizon, freq = 'H')
            data = data.dropna() # No need for this any more

            # prepare learning data
            X = data[all_features].values
            y = data['target'].values

            # fit the model
            self.model.fit(X, y)

            # start evaluation
            # split data to training and testing set
            split = int(X.shape[0] * self.split_point)
            X_train = X[:split]
            y_train = y[:split]
            X_test = X[split:]
            y_test = y[split:]

            # train evaluation model
            evaluation_model = eval(self.algorithm)
            evaluation_model.fit(X_train, y_train)

            with open('performance_rf.txt', 'a+') as data_file:
                data_file.truncate();

                for rec in X_test:
                    start1 = time.time()
                    pred = evaluation_model.predict(rec.reshape(1,-1))
                    end = time.time()
                    latency = end - start1
                    # print(latency)
                    data_file.write("{}\n".format(latency))

            # tesing predictions
            true = y_test
            pred = evaluation_model.predict(X_test)

            # calculate predictability
            fitness = sklearn.metrics.r2_score(true, pred)
            self.predictability = int(max(0, fitness) * 100)

            # calculate evaluation scores
            output = {}
            for metrics in self.err_metrics:
                error_name = metrics['short']
                if error_name == 'rmse':
                    output[error_name] = math.sqrt(sklearn.metrics.mean_squared_error(true, pred))
                else:
                    output[error_name] = metrics['function'](true, pred)
            return output

    def predict(self, ftr_vector, timestamp):
        prediction = self.model.predict(ftr_vector)

        # Retrain stuff
        if(self.retrain_period is not None):
            # Add current ftr_vector to file
            with open(self.train_file_path, 'r') as data_r:
                # get all lines
                lines = data_r.readlines()
                # Create new line and append it
                new_line = "{\"timestamp\": " + str(timestamp) + ", \"ftr_vector\": " + str(ftr_vector[0]) + "}"
                # If not the first line add \n at the beginning
                if(len(lines)!=0):
                    new_line = "\n" + new_line
                lines.append(new_line)

                # Truncate arrays to correct size
                if(self.samples_for_retrain is not None and self.samples_for_retrain < len(lines)):
                   lines = lines[-self.samples_for_retrain:]
                

            with open(self.train_file_path, 'w') as data_w:
                data_w.writelines(lines)
                
            self.samples_from_retrain += 1
            # If conditions are satisfied retrain the model
            if(self.samples_from_retrain%self.retrain_period == 0 and
               (self.samples_for_retrain is None or self.samples_for_retrain == len(lines))):
                self.samples_from_retrain = 0
                self.fit(filename=self.train_file_path)

        return prediction

    def evaluate(self, output, measurement):
        prediction = output['value']
        self.measurements.append(measurement)
        self.predictions.append(prediction)

        # check if buffers are full
        if len(self.predictions) < self.predictions.maxlen:
            warn_text = "Warning: Not enough predictions for evaluation yet ({}/{})".format(len(self.predictions), self.predictions.maxlen)
            warnings.warn(warn_text)
            return output

        true = list(self.measurements)
        pred = list(self.predictions)[:-self.horizon]

        # calculate metrics and append to output
        for metrics in self.err_metrics:
            error_name = metrics['short']
            if error_name == 'rmse':
                output[error_name] = math.sqrt(sklearn.metrics.mean_squared_error(true, pred))
            else:
                output[error_name] = metrics['function'](true, pred)
        return output

    def save(self, filename):
        joblib.dump(self.model, filename, compress=3)
        #print "Saved model to", filename

    def load(self, filename):
        self.model = joblib.load(filename)
        #print "Loaded model from", filename
