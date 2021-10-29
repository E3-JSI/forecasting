import sklearn
import joblib
import pandas as pd
import collections
import math
import warnings
import time
import os
import sys
import lightgbm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

sys.path.insert(0, './lib')
from regression_metrics import mean_absolute_percentage_error, rmse
from torch_network import TorchNetwork


class PredictiveModel:
    """
    Predictive model class is a wrapper for scikit learn regression models and PyTorch.
    ref: http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
    ref: https://pytorch.org/
    """

    def __init__(self,
                 sensor,
                 prediction_horizon,
                 evaluation_period=512,
                 err_metrics=None,
                 split_point=0.8,
                 algorithm='torch',
                 retrain_period=None,
                 samples_for_retrain=None,
                 retrain_file_location=None,
                 time_offset='H',
                 learning_rate=4 * 10 ** -5,
                 batch_size=64,
                 training_rounds=100,
                 num_workers=1,
                 gmm_layer="False",
                 initial_imputer="simple",
                 max_iter=15,
                 n_gmm=5,
                 min_n_gmm=1,
                 max_n_gmm=10,
                 gmm_seed=None,
                 verbose=False,
                 **kwargs):

        self.err_metrics = err_metrics
        if not err_metrics:
            self.err_metrics = [
                {'name': "R2 Score", 'short': "r2", 'function': sklearn.metrics.r2_score},
                {'name': "Mean Absolute Error", 'short': "mae", 'function': sklearn.metrics.mean_absolute_error},
                {'name': "Mean Squared Error", 'short': "mse", 'function': sklearn.metrics.mean_squared_error},
                {'name': "Root Mean Squared Error", 'short': "rmse",
                 'function': rmse},
                {'name': "Mean Absolute Percentage Error", 'short': "mape",
                 'function': mean_absolute_percentage_error}
            ]

        self.algorithm = algorithm
        if algorithm == "torch":
            self.model = TorchNetwork(self)
        else:
            self.model = eval(self.algorithm)
        self.sensor = sensor
        self.horizon = prediction_horizon
        self.eval_period = evaluation_period
        self.split_point = split_point
        self.measurements = collections.deque(maxlen=self.eval_period)
        self.predictions = collections.deque(maxlen=(self.eval_period + self.horizon))
        self.predictability = None
        self.time_offset = time_offset

        # Torch model arguments
        self.encoder_length = 1
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_rounds = training_rounds
        self.num_workers = num_workers

        # GMM layer arguments
        self.gmm_layer = True
        if gmm_layer.lower() == "false":
            self.gmm_layer = False

        if initial_imputer.lower() == 'iterative':
            self.initial_imputer = IterativeImputer(max_iter=max_iter)
        else:
            self.initial_imputer = SimpleImputer()

        self.n_gmm = n_gmm
        self.min_n_gmm = min_n_gmm
        self.max_n_gmm = max_n_gmm
        self.gmm_seed = gmm_seed
        self.verbose = verbose

        # Retrain configurations
        self.samples_for_retrain = samples_for_retrain
        self.retrain_period = retrain_period
        self.samples_from_retrain = 0
        self.samples_in_train_file = 0
        self.retrain_memory = {"timestamp": [], "ftr_vector": []}

        if self.retrain_period is not None:
            # Initialize file
            filename = "{}_{}{}_retrain.json".format(sensor, prediction_horizon, time_offset)
            self.train_file_path = os.path.join(retrain_file_location, filename)
            open(self.train_file_path, "w").close()

    def fit(self, filename):

        with open(filename) as data_file:
            data = pd.read_json(data_file, lines=True)  # if not valid json
            # set datetime as index
            data.set_index('timestamp', inplace=True)
            # transform ftr_vector from array to separate fields
            data = data['ftr_vector'].apply(pd.Series)

            # get features
            all_features = list(data)

            # prepare target based on prediction horizon (first one is measurement to shift)
            measurements = data[[data.columns[0]]]
            data['target'] = measurements.shift(periods=-self.horizon, freq=self.time_offset)
            data = data.dropna()  # No need for this any more

            # prepare learning data
            x = data[all_features].values
            y = data['target'].values

            # fit the model
            self.model.fit(x, y)

            # start evaluation
            # split data to training and testing set
            split = int(x.shape[0] * self.split_point)
            x_train = x[:split]
            y_train = y[:split]
            x_test = x[split:]
            y_test = y[split:]

            # train evaluation model
            if self.algorithm == 'torch':
                evaluation_model = self.TorchNetwork(self)
            else:
                evaluation_model = eval(self.algorithm)
            evaluation_model.fit(x_train, y_train)

            with open('performance_rf.txt', 'a+') as data_file:
                data_file.truncate()

                for rec in x_test:
                    start1 = time.time()
                    evaluation_model.predict(rec.reshape(1, -1))
                    end = time.time()
                    latency = end - start1
                    data_file.write("{}\n".format(latency))

            # testing predictions
            true = y_test
            pred = evaluation_model.predict(x_test)

            # calculate predictability
            fitness = sklearn.metrics.r2_score(true, pred)
            self.predictability = int(max(0, fitness) * 100)

            # calculate evaluation scores
            output = {}
            for metrics in self.err_metrics:
                output[metrics['short']] = metrics['function'](true, pred)
            return output

    def predict(self, ftr_vector, timestamp):
        prediction = self.model.predict(ftr_vector)

        # TODO retraining should be done in asynchronous fashion to prevent long training time to freeze up the whole
        #  process of predictions.
        if self.retrain_period is not None:
            # Add current ftr_vector to file
            with open(self.train_file_path, 'r') as data_r:
                # get all lines
                lines = data_r.readlines()
                # Create new line and append it
                new_line = "{\"timestamp\": " + str(timestamp) + ", \"ftr_vector\": " + str(ftr_vector[0]) + "}"
                # If not the first line add \n at the beginning
                if len(lines) != 0:
                    new_line = "\n" + new_line
                lines.append(new_line)

                # Truncate arrays to correct size
                if self.samples_for_retrain is not None and self.samples_for_retrain < len(lines):
                    lines = lines[-self.samples_for_retrain:]

            with open(self.train_file_path, 'w') as data_w:
                data_w.writelines(lines)

            self.samples_from_retrain += 1
            # If conditions are satisfied retrain the model
            if (self.samples_from_retrain % self.retrain_period == 0 and
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
            warn_text = "Warning: Not enough predictions for evaluation yet ({}/{})".format(len(self.predictions),
                                                                                            self.predictions.maxlen)
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

    def load(self, filename):
        self.model = joblib.load(filename)
