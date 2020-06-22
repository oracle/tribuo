#! /usr/bin/env python3

#  Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# Script for training the xgboost and logistic regression models used in the onnx tests.

import xgboost as xgb
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.linear_model import LogisticRegression


def transpose_and_save_mnist(path_to_mnist: str, output_path: str):
    X, y = load_svmlight_file(path_to_mnist,784)
    inverted = [i for i in range(783,-1,-1)]
    X_trans = X.toarray()[:,inverted]
    with open(output_path, "wb") as f:
        dump_svmlight_file(X_trans,y,f)
    return X_trans, y


def fit_and_save_logistic_regression(X,y,output_path: str):
    lr = LogisticRegression()
    lr.fit(X,y)
    
    initial_type = [('float_input', FloatTensorType([None, 784]))]
    onx = convert_sklearn(lr, initial_types=initial_type)
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())


def fit_and_save_xgb(X,y,output_path:str):
    dtrain = xgb.DMatrix(X, label=y)
    param = {'max_depth': 6, 'eta': 1, 'objective': 'multi:softprob', 'num_class' : 10, 'nthread' : 6}
    bst = xgb.train(param, dtrain, 20)
    bst.save_model(output_path)


if __name__ == "__main__":
    X_trans, y = transpose_and_save_mnist("mnist","transposed_mnist.libsvm")
    X_test_trans, y_test = transpose_and_save_mnist("mnist.t","transposed_mnist_test.libsvm")
    fit_and_save_logistic_regression(X_trans,y,"lr_mnist.onnx")
    fit_and_save_xgb(X_trans,y,"xgb_mnist.xgb")

