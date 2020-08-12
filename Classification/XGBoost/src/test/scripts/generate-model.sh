#!/usr/bin/env bash

NUM_TREES=$1
MODEL_NAME=$2
TRAINDATA=$3
TESTDATA=$4

MAINCLASS="org.tribuo.classification.xgboost.TrainTest"
CLASSPATH=target/tribuo-classification-xgboost-4.0.0-jar-with-dependencies.jar

java -cp $CLASSPATH $MAINCLASS -s TEXT --xgb-ensemble-size ${NUM_TREES} --xgb-num-threads 1 -f ${MODEL_NAME}.model -u $TRAINDATA -v $TESTDATA
