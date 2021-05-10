/*
 * Copyright (c) 2021 Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop.tensorflow;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.framework.optimizers.AdaDelta;
import org.tensorflow.framework.optimizers.AdaGrad;
import org.tensorflow.framework.optimizers.AdaGradDA;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.framework.optimizers.Adamax;
import org.tensorflow.framework.optimizers.Ftrl;
import org.tensorflow.framework.optimizers.GradientDescent;
import org.tensorflow.framework.optimizers.Momentum;
import org.tensorflow.framework.optimizers.Nadam;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.framework.optimizers.RMSProp;
import org.tensorflow.op.Op;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.family.TNumber;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * An enum for the gradient optimisers exposed by TensorFlow-Java.
 */
public enum GradientOptimiser {

    /**
     * The AdaDelta optimiser.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the overall learning rate.</li>
     *     <li>rho - the decay factor.</li>
     *     <li>epsilon - for numerical stability.</li>
     * </ul>
     */
    ADADELTA("learningRate","rho","epsilon"),
    /**
     * The AdaGrad optimiser.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the overall learning rate.</li>
     *     <li>initialAccumulatorValue - the initialisation value for the gradient accumulator.</li>
     * </ul>
     */
    ADAGRAD("learningRate","initialAccumulatorValue"),
    /**
     * The AdaGrad Dual Averaging optimiser.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the overall learning rate.</li>
     *     <li>initialAccumulatorValue - the initialisation value for the gradient accumulator.</li>
     *     <li>l1Strength - the strength of l1 regularisation.</li>
     *     <li>l2Strength - the strength of l2 regularisation.</li>
     * </ul>
     */
    ADAGRADDA("learningRate","initialAccumulatorValue","l1Strength","l2Strength"),
    /**
     * The Adam optimiser.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     *     <li>betaOne - the exponential decay rate for the 1st moment estimates.</li>
     *     <li>betaTwo - the exponential decay rate for the exponentially weighted infinity norm.</li>
     *     <li>epsilon - a small constant for numerical stability.</li>
     * </ul>
     */
    ADAM("learningRate","betaOne","betaTwo","epsilon"),
    /**
     * The Adamax optimiser.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     *     <li>betaOne - the exponential decay rate for the 1st moment estimates.</li>
     *     <li>betaTwo - the exponential decay rate for the exponentially weighted infinity norm.</li>
     *     <li>epsilon - a small constant for numerical stability.</li>
     * </ul>
     */
    ADAMAX("learningRate","betaOne","betaTwo","epsilon"),
    /**
     * The FTRL optimiser.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     *     <li>learningRatePower - controls how the learning rate decreases during training. Use zero for
     *     a fixed learning rate.</li>
     *     <li>initialAccumulatorValue - the starting value for accumulators. Only zero or positive
     *     values are allowed.</li>
     *     <li>l1Strength - the L1 Regularization strength, must be greater than or equal to zero.</li>
     *     <li>l2Strength - the L2 Regularization strength, must be greater than or equal to zero.</li>
     *     <li>l2ShrinkageRegularizationStrength - this differs from L2 above in that the L2 above is a
     *     stabilization penalty, whereas this L2 shrinkage is a magnitude penalty. must be greater
     *     than or equal to zero.</li>
     * </ul>
     */
    FTRL("learningRate","learningRatePower","initialAccumulatorValue","l1Strength","l2Strength","l2ShrinkageRegularizationStrength"),
    /**
     * A standard gradient descent optimiser with a fixed learning rate.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     * </ul>
     */
    GRADIENT_DESCENT("learningRate"),
    /**
     * Gradient descent with momentum.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     *     <li>momentum - the momentum scalar.</li>
     * </ul>
     */
    MOMENTUM("learningRate","momentum"),
    /**
     * Gradient descent with Nesterov momentum.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     *     <li>momentum - the momentum scalar.</li>
     * </ul>
     */
    NESTEROV("learningRate","momentum"),
    /**
     * The Nadam optimiser.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the learning rate.</li>
     *     <li>betaOne - the exponential decay rate for the 1st moment estimates.</li>
     *     <li>betaTwo - the exponential decay rate for the exponentially weighted infinity norm.</li>
     *     <li>epsilon - a small constant for numerical stability.</li>
     * </ul>
     */
    NADAM("learningRate","betaOne","betaTwo","epsilon"),
    /**
     * The RMSprop optimiser.
     * <p>
     * Parameters are:
     * <ul>
     *     <li>learningRate - the overall learning rate.</li>
     *     <li>decay - the decay factor.</li>
     *     <li>momentum - the momentum scalar.</li>
     *     <li>epsilon - for numerical stability.</li>
     * </ul>
     * This optimiser is currently uncentered.
     */
    RMSPROP("learningRate","decay","momentum","epsilon");

    private final Set<String> args;

    /**
     * Construct the enum, storing the optimiser argument names.
     * @param args The optimiser argument names.
     */
    private GradientOptimiser(String... args) {
        this.args = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(args)));
    }

    /**
     * An unmodifiable view of the parameter names used by this gradient optimiser.
     * @return The parameter names.
     */
    public Set<String> getParameterNames() {
        return args;
    }

    /**
     * Checks that the parameter names in the supplied set are an exact
     * match for the parameter names that this gradient optimiser expects.
     * @param paramNames The gradient optimiser parameter names.
     * @return True if the two sets intersection and union are equal.
     */
    public boolean validateParamNames(Set<String> paramNames) {
        return (args.size() == paramNames.size()) && args.containsAll(paramNames);
    }

    /**
     * Applies the optimiser to the graph and returns the optimiser step operation.
     * @param graph The graph to optimise.
     * @param loss The loss to minimise.
     * @param optimiserParams The optimiser parameters.
     * @param <T> The loss type (most of the time this will be {@link TFloat32}.
     * @return The optimiser step operation.
     */
    public <T extends TNumber> Op applyOptimiser(Graph graph, Operand<T> loss, Map<String,Float> optimiserParams) {
        if (!validateParamNames(optimiserParams.keySet())) {
            throw new IllegalArgumentException("Invalid optimiser parameters, expected " + args.toString() + ", found " + optimiserParams.keySet().toString());
        }
        Optimizer optimiser;
        switch (this) {
            case ADADELTA:
                optimiser = new AdaDelta(graph,"tribuo-adadelta",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("rho"),
                        optimiserParams.get("epsilon"));
                break;
            case ADAGRAD:
                optimiser = new AdaGrad(graph,"tribuo-adagrad",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("initialAccumulatorValue"));
                break;
            case ADAGRADDA:
                optimiser = new AdaGradDA(graph,"tribuo-adagradda",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("initialAccumulatorValue"),
                        optimiserParams.get("l1Strength"),
                        optimiserParams.get("l2Strength"));
                break;
            case ADAM:
                optimiser = new Adam(graph,"tribuo-adam",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("betaOne"),
                        optimiserParams.get("betaTwo"),
                        optimiserParams.get("epsilon"));
                break;
            case ADAMAX:
                optimiser = new Adamax(graph,"tribuo-adamax",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("betaOne"),
                        optimiserParams.get("betaTwo"),
                        optimiserParams.get("epsilon"));
                break;
            case FTRL:
                optimiser = new Ftrl(graph,"tribuo-ftrl",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("learningRatePower"),
                        optimiserParams.get("initialAccumulatorValue"),
                        optimiserParams.get("l1Strength"),
                        optimiserParams.get("l2Strength"),
                        optimiserParams.get("l2ShrinkageRegularizationStrength"));
                break;
            case GRADIENT_DESCENT:
                optimiser = new GradientDescent(graph,"tribuo-sgd",
                        optimiserParams.get("learningRate"));
                break;
            case MOMENTUM:
                optimiser = new Momentum(graph,"tribuo-momentum",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("momentum"),
                        false);
                break;
            case NESTEROV:
                optimiser = new Momentum(graph,"tribuo-nesterov",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("momentum"),
                        true);
                break;
            case NADAM:
                optimiser = new Nadam(graph,"tribuo-nadam",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("betaOne"),
                        optimiserParams.get("betaTwo"),
                        optimiserParams.get("epsilon"));
                break;
            case RMSPROP:
                optimiser = new RMSProp(graph,"tribuo-rmsprop",
                        optimiserParams.get("learningRate"),
                        optimiserParams.get("decay"),
                        optimiserParams.get("momentum"),
                        optimiserParams.get("epsilon"),
                        false);
                break;
            default:
                throw new IllegalStateException("Unimplemented switch branch " + this.toString());
        }
        return optimiser.minimize(loss,"tribuo-" + this.toString() + "-minimize");
    }
}
