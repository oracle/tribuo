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

package org.tribuo.classification.sgd.kernel;

import com.oracle.labs.mlrg.olcut.config.ArgumentException;
import com.oracle.labs.mlrg.olcut.config.Option;
import org.tribuo.Trainer;
import org.tribuo.classification.ClassificationOptions;
import org.tribuo.math.kernel.Kernel;
import org.tribuo.math.kernel.Linear;
import org.tribuo.math.kernel.Polynomial;
import org.tribuo.math.kernel.RBF;
import org.tribuo.math.kernel.Sigmoid;

import java.util.logging.Logger;

/**
 * Options for using the KernelSVMTrainer.
 * <p>
 * See:
 * <pre>
 * Shalev-Shwartz S, Singer Y, Srebro N, Cotter A
 * "Pegasos: Primal Estimated Sub-Gradient Solver for SVM"
 * Mathematical Programming, 2011.
 * </pre>
 */
public class KernelSVMOptions implements ClassificationOptions<KernelSVMTrainer> {
    private static final Logger logger = Logger.getLogger(KernelSVMOptions.class.getName());

    /**
     * The kernel types.
     */
    public enum KernelEnum {
        /**
         * Uses a {@link Linear} kernel.
         */
        LINEAR,
        /**
         * Uses a {@link Polynomial} kernel.
         */
        POLYNOMIAL,
        /**
         * Uses a {@link Sigmoid} kernel.
         */
        SIGMOID,
        /**
         * Uses an {@link RBF} kernel.
         */
        RBF
    }

    /**
     * Intercept in kernel function. Defaults to 1.0.
     */
    @Option(longName = "kernel-intercept", usage = "Intercept in kernel function. Defaults to 1.0.")
    public double kernelIntercept = 1.0;
    /**
     * Degree in polynomial kernel function. Defaults to 1.0.
     */
    @Option(longName = "kernel-degree", usage = "Degree in polynomial kernel function. Defaults to 1.0.")
    public double kernelDegree = 1.0;
    /**
     * Gamma value in kernel function. Defaults to 1.0.
     */
    @Option(longName = "kernel-gamma", usage = "Gamma value in kernel function. Defaults to 1.0.")
    public double kernelGamma = 1.0;
    /**
     * Number of SGD epochs. Defaults to 5.
     */
    @Option(longName = "kernel-epochs", usage = "Number of SGD epochs. Defaults to 5.")
    public int kernelEpochs = 5;
    /**
     * Kernel function. Defaults to LINEAR.
     */
    @Option(longName = "kernel-kernel", usage = "Kernel function. Defaults to LINEAR.")
    public KernelEnum kernelKernel = KernelEnum.LINEAR; //TODO should the default be KernelEnum.RBF?
    /**
     * Lambda value in gradient optimisation. Defaults to 0.01.
     */
    @Option(longName = "kernel-lambda", usage = "Lambda value in gradient optimisation. Defaults to 0.01.")
    public double kernelLambda = 0.01;
    /**
     * Log the objective after n examples. Defaults to 100.
     */
    @Option(longName = "kernel-logging-interval", usage = "Log the objective after <int> examples. Defaults to 100.")
    public int kernelLoggingInterval = 100;
    /**
     * Sets the random seed for the Kernel SVM.
     */
    @Option(longName = "kernel-seed", usage = "Sets the random seed for the Kernel SVM.")
    public long kernelSeed = Trainer.DEFAULT_SEED;

    @Override
    public KernelSVMTrainer getTrainer() {
        logger.info("Configuring Kernel SVM Trainer");
        Kernel kernelObj = null;
        switch (kernelKernel) {
            case LINEAR:
                logger.info("Using a linear kernel");
                kernelObj = new Linear();
                break;
            case POLYNOMIAL:
                logger.info("Using a Polynomial kernel with gamma " + kernelGamma + ", intercept " + kernelIntercept + ", and degree " + kernelDegree);
                kernelObj = new Polynomial(kernelGamma, kernelIntercept, kernelDegree);
                break;
            case RBF:
                logger.info("Using an RBF kernel with gamma " + kernelGamma);
                kernelObj = new RBF(kernelGamma);
                break;
            case SIGMOID:
                logger.info("Using a tanh kernel with gamma " + kernelGamma + ", and intercept " + kernelIntercept);
                kernelObj = new Sigmoid(kernelGamma, kernelIntercept);
                break;
            default:
                logger.warning("Unknown kernel function " + kernelKernel);
                throw new ArgumentException("kernel-kernel", "Unknown kernel function " + kernelKernel);
        }
        logger.info(String.format("Set logging interval to %d", kernelLoggingInterval));
        return new KernelSVMTrainer(kernelObj, kernelLambda, kernelEpochs, kernelLoggingInterval, kernelSeed);
    }
}
