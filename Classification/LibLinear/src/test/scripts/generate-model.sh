#!/usr/bin/env bash

LLTYPE=$1
MODEL_NAME=$2
TRAINDATA=$3
TESTDATA=$4

MAINCLASS="org.tribuo.classification.liblinear.TrainTest"
CLASSPATH=target/tribuo-classification-liblinear-4.0.0-jar-with-dependencies.jar

java -cp $CLASSPATH $MAINCLASS -s TEXT --liblinear-solver-type ${LLTYPE} -f ${MODEL_NAME}.model -u $TRAINDATA -v $TESTDATA
