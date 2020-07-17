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

/**
 * An implementation of the Adam gradient optimiser.
 * <p>
 * Creates two copies of the parameters to store learning rates.
 * <p>
 * See:
 * <pre>
 * Kingma, D., and Ba, J.
 * "Adam: A Method for Stochastic Optimization"
 * arXiv preprint arXiv:1412.6980, 2014.
 * </pre>
 */
public class Adam implements StochasticGradientOptimiser {

    @Config(description="Learning rate to scale the gradients by.")
    private double initialLearningRate = 0.001;

    @Config(description="The beta one parameter.")
    private double betaOne = 0.9;

    @Config(description="The beta two parameter.")
    private double betaTwo = 0.99;

    @Config(description="Epsilon for numerical stability.")
    private double epsilon = 1e-6;

    private int iterations = 0;
    private Tensor[] firstMoment;
    private Tensor[] secondMoment;

    /**
     * It's highly recommended not to modify these parameters, use one of the
     * other constructors.
     * @param initialLearningRate The initial learning rate.
     * @param betaOne The value of beta-one.
     * @param betaTwo The value of beta-two.
     * @param epsilon The epsilon value.
     */
    public Adam(double initialLearningRate, double betaOne, double betaTwo, double epsilon) {
        this.initialLearningRate = initialLearningRate;
        this.betaOne = betaOne;
        this.betaTwo = betaTwo;
        this.epsilon = epsilon;
        this.iterations = 0;
    }

    /**
     * Sets betaOne to 0.9 and betaTwo to 0.999
     * @param initialLearningRate The initial learning rate.
     * @param epsilon The epsilon value.
     */
    public Adam(double initialLearningRate, double epsilon) {
        this(initialLearningRate,0.9,0.999,epsilon);
    }

    /**
     * Sets initialLearningRate to 0.001, betaOne to 0.9, betaTwo to 0.999, epsilon to 1e-6.
     * These are the parameters from the Adam paper.
     */
    public Adam() {
        this(0.001,0.9,0.999,1e-6);
    }

    @Override
    public void initialise(Parameters parameters) {
        firstMoment = parameters.getEmptyCopy();
        secondMoment = parameters.getEmptyCopy();
        iterations = 0;
    }

    @Override
    public Tensor[] step(Tensor[] updates, double weight) {
        iterations++;

        double learningRate = initialLearningRate * Math.sqrt(1.0 - Math.pow(betaTwo,iterations)) / (1.0 - Math.pow(betaOne,iterations));
        //lifting lambdas out of the for loop until JDK-8183316 is fixed.
        DoubleUnaryOperator scale = (double a) -> a * learningRate;

        for (int i = 0; i < updates.length; i++) {
            firstMoment[i].scaleInPlace(betaOne);
            firstMoment[i].intersectAndAddInPlace(updates[i],(double a) -> a * (1.0 - betaOne));
            secondMoment[i].scaleInPlace(betaTwo);
            secondMoment[i].intersectAndAddInPlace(updates[i],(double a) -> a * a * (1.0 - betaTwo));
            updates[i].scaleInPlace(0.0); //scales everything to zero, but leaving the sparse presence
            updates[i].intersectAndAddInPlace(firstMoment[i],scale); // add in the first moment
            updates[i].hadamardProductInPlace(secondMoment[i],(double a) -> Math.sqrt(a) + epsilon); // scale by second moment
        }

        return updates;
    }

    @Override
    public String toString() {
        return "Adam(learningRate="+initialLearningRate+",betaOne="+betaOne+",betaTwo="+betaTwo+",epsilon="+epsilon+")";
    }

    @Override
    public void reset() {
        firstMoment = null;
        secondMoment = null;
        iterations = 0;
    }

    @Override
    public Adam copy() {
        return new Adam(initialLearningRate,betaOne,betaTwo,epsilon);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"StochasticGradientOptimiser");
    }
}
