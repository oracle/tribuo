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

package org.tribuo.math.optimisers;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.math.Parameters;
import org.tribuo.math.StochasticGradientOptimiser;
import org.tribuo.math.la.Tensor;

import java.util.function.DoubleUnaryOperator;
import java.util.logging.Logger;

/**
 * An implementation of the AdaGrad gradient optimiser.
 * <p>
 * Creates one copy of the parameters to store learning rates.
 * <p>
 * See:
 * <pre>
 * Duchi, J., Hazan, E., and Singer, Y.
 * "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
 * Journal of Machine Learning Research, 2012, 2121-2159.
 * </pre>
 */
public class AdaGrad implements StochasticGradientOptimiser {
    private static final Logger logger = Logger.getLogger(AdaGrad.class.getName());

    @Config(mandatory = true,description="Initial learning rate used to scale the gradients.")
    private double initialLearningRate;

    @Config(description="Epsilon for numerical stability around zero.")
    private double epsilon = 1e-6;

    @Config(description="Initial value for the gradient accumulator.")
    private double initialValue = 0.0;

    private Tensor[] gradsSquared;

    /**
     * Creates an AdaGrad optimiser using the specified learning rate, epsilon and initial accumulator value.
     * @param initialLearningRate The learning rate.
     * @param epsilon The epsilon value for stabilising the gradient inversion.
     * @param initialValue The initial value for the gradient accumulator.
     */
    public AdaGrad(double initialLearningRate, double epsilon, double initialValue) {
        this.initialLearningRate = initialLearningRate;
        this.epsilon = epsilon;
        this.initialValue = initialValue;
    }

    /**
     * Creates an AdaGrad optimiser using the specified learning rate and epsilon.
     * <p>
     * Sets the initial value for the accumulator to zero.
     * @param initialLearningRate The learning rate.
     * @param epsilon The epsilon value for stabilising the gradient inversion.
     */
    public AdaGrad(double initialLearningRate, double epsilon) {
        this(initialLearningRate,epsilon,0.0);
    }

    /**
     * Creates an AdaGrad optimiser using the specified initial learning rate.
     * <p>
     * Sets epsilon to 1e-6, and the initial accumulator value to zero.
     * @param initialLearningRate The learning rate.
     */
    public AdaGrad(double initialLearningRate) {
        this(initialLearningRate,1e-6,0.0);
    }

    /**
     * For OLCUT.
     */
    private AdaGrad() { }

    @Override
    public void initialise(Parameters parameters) {
        this.gradsSquared = parameters.getEmptyCopy();
        if (initialValue != 0.0) {
            for (Tensor t : gradsSquared) {
                t.scalarAddInPlace(initialValue);
            }
        }
    }

    @Override
    public Tensor[] step(Tensor[] updates, double weight) {
        //lifting lambdas out of the for loop until JDK-8183316 is fixed.
        DoubleUnaryOperator square = (double a) -> weight*weight*a*a;
        DoubleUnaryOperator scale = (double a) -> weight * initialLearningRate / (epsilon + Math.sqrt(a));
        for (int i = 0; i < updates.length; i++) {
            Tensor curGradsSquared = gradsSquared[i];
            Tensor curGrad = updates[i];
            curGradsSquared.intersectAndAddInPlace(curGrad,square);
            curGrad.hadamardProductInPlace(curGradsSquared,scale);
        }

        return updates;
    }

    @Override
    public String toString() {
        return "AdaGrad(initialLearningRate="+initialLearningRate+",epsilon="+epsilon+",initialValue="+initialValue+")";
    }

    @Override
    public void reset() {
        gradsSquared = null;
    }

    @Override
    public AdaGrad copy() {
        return new AdaGrad(initialLearningRate,epsilon);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"StochasticGradientOptimiser");
    }
}
