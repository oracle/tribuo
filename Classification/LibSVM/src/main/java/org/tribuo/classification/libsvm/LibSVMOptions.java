/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tribuo.classification.libsvm;

import com.oracle.labs.mlrg.olcut.config.Option;
import org.tribuo.classification.ClassificationOptions;
import org.tribuo.classification.Label;
import org.tribuo.classification.libsvm.SVMClassificationType.SVMMode;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.SVMParameters;

/**
 * CLI options for training a LibSVM classification model.
 */
public class LibSVMOptions implements ClassificationOptions<LibSVMClassificationTrainer> {

    @Override
    public String getOptionsDescription() {
        return "Options for parameterising a LibSVM classification trainer.";
    }

    /**
     * Intercept in kernel function. Defaults to 0.0.
     */
    @Option(longName = "svm-coefficient", usage = "Intercept in kernel function. Defaults to 0.0.")
    public double svmCoefficient = 0.0;  //TODO should this be 1.0?
    /**
     * Degree in polynomial kernel. Defaults to 3.
     */
    @Option(longName = "svm-degree", usage = "Degree in polynomial kernel. Defaults to 3.")
    public int svmDegree = 3;
    /**
     * Gamma value in kernel function. Defaults to 0.0.
     */
    @Option(longName = "svm-gamma", usage = "Gamma value in kernel function. Defaults to 0.0.")
    public double svmGamma = 0.0;  //TODO should the default be 0.1
    /**
     * Type of SVM kernel. Defaults to LINEAR.
     */
    @Option(longName = "svm-kernel", usage = "Type of SVM kernel. Defaults to LINEAR.")
    public KernelType svmKernel = KernelType.LINEAR;
    /**
     * Type of SVM. Defaults to C_SVC.
     */
    @Option(longName = "svm-type", usage = "Type of SVM. Defaults to C_SVC.")
    public SVMClassificationType.SVMMode svmType = SVMMode.C_SVC;

    @Override
    public LibSVMClassificationTrainer getTrainer() {
        SVMParameters<Label> parameters = new SVMParameters<>(new SVMClassificationType(svmType), svmKernel);
        parameters.setGamma(svmGamma);
        parameters.setCoeff(svmCoefficient);
        parameters.setDegree(svmDegree);
        return new LibSVMClassificationTrainer(parameters);
    }

}