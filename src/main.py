#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
import json
import time
import os.path
import sklearn
import threading
import requests

# adding lib subdirectory
sys.path.insert(0,'./lib')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import joblib
import regression_metrics as additional_metrics

from kafka import KafkaConsumer
from kafka import KafkaProducer
import numpy as np
import pandas as pd
from predictive_model import PredictiveModel

def get_model_file_name(sensor, horizon):
    subdir = 'models'
    if not os.path.isdir(subdir):
        os.makedirs(subdir)

    filename = "model_{}_{}h".format(sensor, horizon)
    filepath = os.path.join(subdir, filename)

    return filepath

def get_data_file_name(sensor, horizon):
    subdir = '../../data/fused'
    if not os.path.isdir(subdir):
        os.makedirs(subdir)

    filename = "{}_{}h.json".format(sensor, horizon)
    filepath = os.path.join(subdir, filename)

    return filepath

def get_input_data_topics(sensors, horizons):
    topics = []
    for sensor in sensors:
        for horizon in horizons:
            topics.append("features_{}_{}h".format(sensor, horizon))

    return topics

def ping_watchdog():
    interval = 60 # ping interval in seconds
    url = "localhost"
    port= 3001
    path= "/ping?id=5&secret=b9347c25aba4d3ba6e8f61d05fd1c011"

    try:
        r = requests.get("http://{}:{}{}".format(url, port, path))
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print(e)
    else:
        print('Successful ping at ' + time.ctime())

    threading.Timer(interval, ping_watchdog).start()

def main():
    parser = argparse.ArgumentParser(description="Modeling component")

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config.json",
        help=u"Config file located in ./config/ directory",
    )

    parser.add_argument(
        "-f",
        "--fit",
        action='store_true',
        help=u"Learning the model from dataset in subfolder '../../data/fused'",
    )

    parser.add_argument(
        "-s",
        "--save",
        action='store_true',
        help=u"Saving models to subfolder '/models'"
    )

    parser.add_argument(
        "-l",
        "--load",
        action='store_true',
        help=u"Loading models from subfolder '/models'"
    )

    parser.add_argument(
        "-p",
        "--predict",
        dest="predict",
        action='store_true',
        help=u"Start live predictions",
    )

    parser.add_argument(
        "-w",
        "--watchdog",
        dest="watchdog",
        action='store_true',
        help=u"Ping watchdog",
    )

    # Display help if no arguments are defined
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    # Parse input arguments
    args = parser.parse_args()

    #Read config file
    with open("config/" + args.config) as data_file:
        conf = json.load(data_file)

    # Initialize models
    print("\n=== Init phase ===")

    models = {}
    algorithm = conf['algorithm']
    sensors = conf['sensors']
    horizons = conf['prediction_horizons']
    evaluation_period = conf['evaluation_period']
    evaluation_split_point = conf['evaluation_split_point']
    error_metrics = [
        {'name': "R2 Score", 'short': "r2", 'function': sklearn.metrics.r2_score},
        {'name': "Mean Absolute Error", 'short': "mae", 'function': sklearn.metrics.mean_absolute_error},
        {'name': "Mean Squared Error", 'short': "mse", 'function': sklearn.metrics.mean_squared_error},
        {'name': "Root Mean Squared Error", 'short': "rmse", 'function': None},
        {'name': "Mean Absolute Percentage Error", 'short': "mape", 'function': additional_metrics.mean_absolute_percentage_error}
    ]

    # Ifretrain period is defined read it from conf otherwise use None
    if("retrain_period" in conf):
        retrain_period = conf["retrain_period"]
        if("samples_for_retrain" in conf):
            samples_for_retrain = conf["samples_for_retrain"]
        else:
            samples_for_retrain = None
    else:
        retrain_period = None
        samples_for_retrain = None

    for sensor in sensors:
        models[sensor] = {}
        for horizon in horizons:
            models[sensor][horizon] = PredictiveModel(algorithm, sensor,
                                                      horizon,
                                                      evaluation_period,
                                                      error_metrics,
                                                      evaluation_split_point,
                                                      retrain_period,
                                                      samples_for_retrain,
                                                      os.path.join('.', 'test', 'retrain_data'))
            print("Initializing model_{}_{}h".format(sensor, horizon))

    # Model learning
    if (args.fit):
        print("\n=== Learning phase ===")

        for sensor in sensors:
            for horizon in horizons:
                start = time.time()
                data = get_data_file_name(sensor, horizon)
                try:
                    score = models[sensor][horizon].fit(data)
                except Exception as e:
                    print(e)
                end = time.time()
                print("Model[{0}_{1}h] training time: {2:.1f}s, evaluations: {3})".format(sensor, horizon, end-start, str(score)))

    # Model saving
    if (args.save):
        print("\n=== Saving phase ===")

        for sensor in sensors:
            for horizon in horizons:
                model = models[sensor][horizon]
                filename = get_model_file_name(sensor, horizon)
                model.save(filename)
                print("Saved model", filename)

    # Model loading
    if (args.load):
        print("\n=== Loading phase ===")

        for sensor in sensors:
            for horizon in horizons:
                model = models[sensor][horizon]
                filename = get_model_file_name(sensor, horizon)
                model.load(filename)
                print("Loaded model", filename)

    if (args.watchdog):
        print("\n=== Watchdog started ===")
        ping_watchdog()

    # Live predictions
    if (args.predict):
        print("\n=== Predictions phase ===")

        # Start Kafka consumer
        topics = get_input_data_topics(sensors, horizons)
        consumer = KafkaConsumer(bootstrap_servers=conf['bootstrap_servers'])
        consumer.subscribe(topics)
        print("Subscribed to topics: ", topics)

        # Start Kafka producer
        producer = KafkaProducer(bootstrap_servers=conf['bootstrap_servers'],
                                 value_serializer=lambda v: json.dumps(v).encode('utf-8'))

        for msg in consumer:
            try:
                rec = eval(msg.value)
                timestamp = rec['timestamp']
                ftr_vector = rec['ftr_vector']
                measurement = ftr_vector[0] # first feature is the target measurement

                topic = msg.topic

                # extract sensor and horizon info from topic name
                horizon = int(topic.split("_")[-1][:-1])
                sensor = topic.split("_")[-2]

                # predictions
                model = models[sensor][horizon]
                predictions = model.predict([ftr_vector], timestamp)

                # output record
                output = {'stampm': timestamp,
                          'value': predictions[0],
                          'sensor_id': sensor,
                          'horizon': horizon,
                          'predictability': model.predictability}

                # evaluation
                output = model.evaluate(output, measurement) # appends evaluations to output

                # send result to kafka topic
                output_topic = "predictions_{}".format(sensor)
                future = producer.send(output_topic, output)

                print(output_topic + ": " + str(output))

                try:
                    record_metadata = future.get(timeout=10)
                except Exception as e:
                    print('Producer error: ' + str(e))

            except Exception as e:
                print('Consumer error: ' + str(e))

if __name__ == '__main__':
    main()
