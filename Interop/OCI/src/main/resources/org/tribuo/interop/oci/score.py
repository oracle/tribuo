# coding: utf-8
# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is available under the Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0.

import onnxruntime as rt
import os

model_name = 'model.onnx'

"""
    This script is used to serve ONNX models deployed from Java by Tribuo.
"""

def load_model(model_file_name=model_name):
    """
    Loads model from the serialized format
    Returns
    -------
    model:  an onnxruntime session instance
    """
    model_dir = os.path.dirname(os.path.realpath(__file__))
    contents = os.listdir(model_dir)
    if model_file_name in contents:
        return rt.InferenceSession(os.path.join(model_dir, model_file_name))
    else:
        raise Exception('{0} is not found in model directory {1}'.format(model_file_name, model_dir))


def predict(data, model=load_model()):
    """
    Returns prediction given the model and data to predict
    Parameters
    ----------
    model: Model session instance returned by load_model API
    data: Data format as expected by the onnxruntime API
    Returns
    -------
    predictions: Output from scoring server
        Format: {'prediction':output from model.predict method}
    """
    input_data = {'input': data}

    pred = model.run(None, input_data)[0].tolist()
    return {'prediction': pred}
