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
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.optimisers.util.ShrinkingMatrix;
import org.tribuo.math.optimisers.util.ShrinkingVector;

import java.util.function.DoubleUnaryOperator;
import java.util.logging.Logger;

/**
 * An implementation of the RMSProp gradient optimiser.
 * <p>
 * Creates one copy of the parameters to store learning rates.
 * Follows the Keras implementation.
 * <p>
 * See:
 * <pre>
 * Tieleman, T. and Hinton, G.
 * Lecture 6.5 - RMSProp, COURSERA: Neural Networks for Machine Learning.
 * Technical report, 2012.
 * </pre>
 */
public class RMSProp implements StochasticGradientOptimiser {
    private static final Logger logger = Logger.getLogger(RMSProp.class.getName());

    @Config(mandatory = true,description="Learning rate to scale the gradients by.")
    private double initialLearningRate;

    @Config(description="Momentum parameter.")
    private double rho = 0.9;

    @Config(description="Epsilon for numerical stability.")
    private double epsilon = 1e-8;

    @Config(description="Decay factor for the momentum.")
    private double decay = 0.0;

    private double invRho;

    private int iteration = 0;

    private Tensor[] gradsSquared;

    private DoubleUnaryOperator square;

    /**
     * Constructs an RMSProp gradient optimiser using the specified parameters.
     * @param initialLearningRate The initial learning rate.
     * @param rho The momentum parameter.
     * @param epsilon The epsilon to ensure division stability.
     * @param decay The decay parameter.
     */
    public RMSProp(double initialLearningRate, double rho, double epsilon, double decay) {
        this.initialLearningRate = initialLearningRate;
        this.rho = rho;
        this.epsilon = epsilon;
        this.decay = decay;
        this.iteration = 0;
        postConfig();
    }

    /**
     * Constructs an RMSProp gradient optimiser using the specified parameters with epsilon set to 1e-8 and decay to 0.0.
     * @param initialLearningRate The initial learning rate.
     * @param rho The momentum parameter.
     */
    public RMSProp(double initialLearningRate, double rho) {
        this(initialLearningRate,rho,1e-8,0.0);
    }

    /**
     * For olcut.
     */
    private RMSProp() { }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        this.invRho = 1.0 - rho;
        this.square = (double a) -> invRho*a*a;
    }

    @Override
    public void initialise(Parameters parameters) {
        gradsSquared = parameters.getEmptyCopy();
        for (int i = 0; i < gradsSquared.length; i++) {
            if (gradsSquared[i] instanceof DenseVector) {
                gradsSquared[i] = new ShrinkingVector(((DenseVector) gradsSquared[i]), invRho, false);
            } else if (gradsSquared[i] instanceof DenseMatrix) {
                gradsSquared[i] = new ShrinkingMatrix(((DenseMatrix) gradsSquared[i]), invRho, false);
            } else {
                throw new IllegalStateException("Unknown Tensor subclass");
            }
        }
    }

    @Override
    public Tensor[] step(Tensor[] updates, double weight) {
        double learningRate = initialLearningRate / (1 + decay * iteration);
        //lifting lambdas out of the for loop until JDK-8183316 is fixed.
        DoubleUnaryOperator scale = (double a) -> weight * learningRate / (epsilon + Math.sqrt(a));
        for (int i = 0; i < updates.length; i++) {
            Tensor curGradsSquared = gradsSquared[i];
            Tensor curGrad = updates[i];
            curGradsSquared.intersectAndAddInPlace(curGrad,square);
            curGrad.hadamardProductInPlace(curGradsSquared,scale);
        }

        iteration++;
        return updates;
    }

    @Override
    public String toString() {
        return "RMSProp(initialLearningRate="+initialLearningRate+",rho="+rho+",epsilon="+epsilon+",decay="+decay+")";
    }

    @Override
    public void reset() {
        gradsSquared = null;
        iteration = 0;
    }

    @Override
    public RMSProp copy() {
        return new RMSProp(initialLearningRate,rho,epsilon,decay);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"StochasticGradientOptimiser");
    }
}
