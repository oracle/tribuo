#!/usr/bin/env bash

LSTYPE=$1
KTYPE=$2
MODEL_NAME=$3
TRAINDATA=$4
TESTDATA=$5

MAINCLASS="org.tribuo.classification.libsvm.TrainTest"
CLASSPATH=target/tribuo-classification-libsvm-4.0.0-jar-with-dependencies.jar

java -cp $CLASSPATH $MAINCLASS -s TEXT --svm-type ${LSTYPE} --svm-kernel ${KTYPE} -f ${MODEL_NAME}.model -u ${TRAINDATA} -v ${TESTDATA}
