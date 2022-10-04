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

package org.tribuo.common.libsvm;

import org.tribuo.Output;
import libsvm.svm_parameter;

import java.io.Serializable;
import java.util.Arrays;
import java.util.logging.Logger;

/**
 * A container for SVM parameters and the kernel.
 */
public class SVMParameters<T extends Output<T>> implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private static final Logger logger = Logger.getLogger(SVMParameters.class.getName());

    /**
     * The type of the SVM.
     */
    protected final SVMType<T> svmType;

    /**
     * The kernel.
     */
    protected final KernelType kernelType;

    /**
     * The libSVM format parameters.
     */
    protected final svm_parameter parameters = new svm_parameter();

    /**
     * Constructs the default SVMParameters using the specified SVMType and KernelType.
     * @param svmType The SVM algorithm.
     * @param kernelType The kernel.
     */
    public SVMParameters(SVMType<T> svmType, KernelType kernelType) {
        this.svmType = svmType;
        this.kernelType = kernelType;
        parameters.svm_type = svmType.getNativeType();
        parameters.kernel_type = kernelType.getNativeType();
        //
        // These are defaults, which are only compatible with SVM type
        // C_SVC and kernel type RBF
        parameters.degree = 3;
        parameters.gamma = 0;	// 1/num_features
        parameters.coef0 = 0;
        parameters.nu = 0.5;
        parameters.cache_size = 500;
        parameters.C = 1;
        parameters.eps = 1e-3;
        parameters.p = 0.1;
        parameters.shrinking = 1;
        parameters.probability = 0;
        parameters.nr_weight = 0;
        parameters.weight_label = new int[0];
        parameters.weight = new double[0];
    }

    /**
     * Gets the SVM type.
     * @return The SVM type.
     */
    public SVMType<T> getSvmType() {
        return svmType;
    }

    /**
     * Gets the kernel type.
     * @return The kernel type.
     */
    public KernelType getKernelType() {
        return kernelType;
    }

    /**
     * Gets the underlying SVM parameter object.
     * @return The SVM parameters.
     */
    public svm_parameter getParameters() {
        return parameters;
    }

    @Override
    public String toString() {
        return svmParamsToString(parameters);
    }
    
    /**
     * Makes the model that is built provide probability estimates.
     */
    public void setProbability() {
        parameters.probability = 1;
    }

    /**
     * Sets the cost for C_SVC.
     * @param c The cost.
     */
    public void setCost(double c) {
        if(svmType.isNu() || !svmType.isClassification()) {
            logger.warning(String.format("Setting cost %f for non-C_SVC model", c));
        }
        parameters.C = c;
    }

    /**
     * Sets the value of nu for NU_SVM.
     * @param nu The nu.
     */
    public void setNu(double nu) {
        if(!svmType.isNu()) {
            logger.warning(String.format("Setting nu %f for non-NU_SVM model", nu));
        }
        parameters.nu = nu;
    }

    /**
     * Sets the coefficient.
     * @param coeff The coefficient.
     */
    public void setCoeff(double coeff) {
        parameters.coef0 = coeff;
    }

    /**
     * Sets the termination closeness.
     * @param epsilon The termination criterion.
     */
    public void setEpsilon(double epsilon) {
        parameters.p = epsilon;
    }

    /**
     * Sets the degree of the polynomial kernel.
     * @param degree The polynomial degree.
     */
    public void setDegree(int degree) {
        parameters.degree = degree;
    }

    /**
     * Sets gamma in the RBF kernel.
     * @param gamma The gamma.
     */
    public void setGamma(double gamma) {
        parameters.gamma = gamma;
    }

    /**
     * Gets the gamma value.
     * @return The gamma value.
     */
    public double getGamma() {
        return parameters.gamma;
    }

    /**
     * Sets the cache size.
     * @param cacheMB The cache size.
     */
    public void setCacheSize(double cacheMB) {
        if(cacheMB <= 0) {
            throw new IllegalArgumentException("Cache must be larger than 0MB");
        }
        parameters.cache_size = cacheMB;
    }

    /**
     * Deep copy of the svm_parameters including the arrays.
     * @param input The parameters to copy.
     * @return A copy of the svm_parameters.
     */
    public static svm_parameter copyParameters(svm_parameter input) {
        svm_parameter copy = new svm_parameter();
        copy.svm_type = input.svm_type;
        copy.kernel_type = input.kernel_type;
        copy.degree = input.degree;
        copy.gamma = input.gamma;
        copy.coef0 = input.coef0;
        copy.cache_size = input.cache_size;
        copy.eps = input.eps;
        copy.C = input.C;
        copy.nr_weight = input.nr_weight;
        copy.nu = input.nu;
        copy.p = input.p;
        copy.shrinking = input.shrinking;
        copy.probability = input.probability;
        copy.weight_label = input.weight_label != null ? Arrays.copyOf(input.weight_label,input.weight_label.length) : null;
        copy.weight = input.weight != null ? Arrays.copyOf(input.weight,input.weight.length) : null;
        return copy;
    }

    /**
     * A sensible toString for svm_parameter.
     * @param param The parameters.
     * @return A String describing the parameters.
     */
    public static String svmParamsToString(svm_parameter param) {
        StringBuilder sb = new StringBuilder();
        sb.append("svm_parameter(svm_type=");
        sb.append(param.svm_type);
        sb.append(", kernel_type=");
        sb.append(param.kernel_type);
        sb.append(", degree=");
        sb.append(param.degree);
        sb.append(", gamma=");
        sb.append(param.gamma);
        sb.append(", coef0=");
        sb.append(param.coef0);
        sb.append(", cache_size=");
        sb.append(param.coef0);
        sb.append(", eps=");
        sb.append(param.eps);
        sb.append(", C=");
        sb.append(param.C);
        sb.append(", nr_weight=");
        sb.append(param.nr_weight);
        if (param.weight_label != null) {
            sb.append(", weight_label=");
            sb.append(Arrays.toString(param.weight_label));
        }
        if (param.weight != null) {
            sb.append(", weight=");
            sb.append(Arrays.toString(param.weight));
        }
        sb.append(", nu=");
        sb.append(param.nu);
        sb.append(", p=");
        sb.append(param.p);
        sb.append(", shrinking=");
        sb.append(param.shrinking);
        sb.append(", probability=");
        sb.append(param.probability);
        sb.append(')');
        return sb.toString();
    }
}
