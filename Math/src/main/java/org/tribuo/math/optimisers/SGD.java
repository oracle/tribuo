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
 * An implementation of single learning rate SGD and optionally momentum.
 * <p>
 * Has factory methods to generate constant learning rate, linear decay and sqrt decay variants.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 * and for the momentum implementation:
 * <pre>
 * Shallue et al,
 * "Measuring the Effects of Data Parallelism on Neural Network Training"
 * 2018, Arxiv 1811.03600
 * </pre>
 */
public abstract class SGD implements StochasticGradientOptimiser {
    private static final Logger logger = Logger.getLogger(SGD.class.getName());

    /**
     * Momentum types.
     */
    public enum Momentum {
        /**
         * No momentum.
         */
        NONE,
        /**
         * Standard momentum.
         */
        STANDARD,
        /**
         * Nesterov momentum.
         */
        NESTEROV
    }

    /**
     * The initial learning rate.
     */
    @Config(mandatory = true,description="Initial learning rate.")
    protected double initialLearningRate;

    /**
     * Should it use momentum.
     */
    @Config(mandatory = true,description="Momentum type to use.")
    protected Momentum useMomentum;

    /**
     * The scaling factor for the momentum.
     */
    @Config(description="Momentum scaling factor.")
    protected double rho = 0.0;

    /**
     * The iteration number, in steps.
     */
    protected int iteration = 0;

    private Tensor[] momentum;

    SGD(double learningRate) {
        this(learningRate,0.0,Momentum.NONE);
    }

    SGD(double learningRate, double rho, Momentum useMomentum) {
        this.initialLearningRate = learningRate;
        this.useMomentum = useMomentum;
        this.rho = rho;
    }

    /**
     * For olcut.
     */
    protected SGD() { }

    @Override
    public void initialise(Parameters parameters) {
        if (useMomentum != Momentum.NONE) {
            momentum = parameters.getEmptyCopy();
        }
    }

    @Override
    public Tensor[] step(Tensor[] updates, double weight) {
        iteration++;
        double learningRate = learningRate();
        DoubleUnaryOperator learningRateFunc = (double a) -> a * learningRate * weight;
        DoubleUnaryOperator nesterovFunc = (double a) -> a * learningRate * weight * rho;

        /* Modelled after momentum as described in
         * "Measuring the Effects of Data Parallelism on Neural Network Training"
         * Shallue et al 2018, Arxiv 1811.03600
         */
        for (int i = 0; i < updates.length; i++) {
            switch (useMomentum) {
                case STANDARD:
                    momentum[i].scaleInPlace(rho);
                    momentum[i].intersectAndAddInPlace(updates[i]);
                    updates[i].scaleInPlace(0.0);
                    updates[i].intersectAndAddInPlace(momentum[i],learningRateFunc);
                    break;
                case NESTEROV:
                    momentum[i].scaleInPlace(rho);
                    momentum[i].intersectAndAddInPlace(updates[i]);
                    updates[i].scaleInPlace(weight * learningRate);
                    updates[i].intersectAndAddInPlace(momentum[i],nesterovFunc);
                    break;
                case NONE:
                default:
                    updates[i].scaleInPlace(weight * learningRate);
                    break;
            }
        }

        return updates;
    }

    /**
     * Override to provide a function which calculates the learning rate.
     * The only available information is the iteration count.
     * @return The current learning rate.
     */
    public abstract double learningRate();

    /**
     * Override to specify the kind of SGD.
     * @return A string representing the SGD type.
     */
    protected abstract String sgdType();

    @Override
    public String toString() {
        switch (useMomentum) {
            case STANDARD:
                return "SGD+Momentum(type=" + sgdType() + ",initialLearningRate=" + initialLearningRate + ",rho="+rho+")";
            case NESTEROV:
                return "SGD+NesterovMomentum(type=" + sgdType() + ",initialLearningRate=" + initialLearningRate + ",rho="+rho+")";
            default:
                return "SGD(type=" + sgdType() + ",initialLearningRate=" + initialLearningRate + ")";
        }
    }

    @Override
    public void reset() {
        momentum = null;
        iteration = 0;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"StochasticGradientOptimiser");
    }

    /**
     * Generates an SGD optimiser with a constant learning rate set to learningRate.
     * @param learningRate The learning rate.
     * @return A constant learning rate SGD.
     */
    public static SGD getSimpleSGD(double learningRate) {
        return new SimpleSGD(learningRate);
    }

    /**
     * Generates an SGD optimiser with a constant learning rate set to learningRate, with momentum.
     * @param learningRate The learning rate.
     * @param rho The momentum drag constant.
     * @param momentumType Momentum type.
     * @return A constant learning rate SGD with momentum.
     */
    public static SGD getSimpleSGD(double learningRate, double rho, Momentum momentumType) {
        return new SimpleSGD(learningRate, rho, momentumType);
    }

    /**
     * Generates an SGD optimiser with a linearly decaying learning rate initialised to learningRate.
     * <p>
     * The learning rate = initialLearningRate / iteration.
     * @param learningRate The learning rate.
     * @return A linear decay SGD.
     */
    public static SGD getLinearDecaySGD(double learningRate) {
        return new LinearDecaySGD(learningRate);
    }

    /**
     * Generates an SGD optimiser with a linearly decaying learning rate initialised to learningRate, with momentum.
     * <p>
     * The learning rate = initialLearningRate / iteration.
     * @param learningRate The learning rate.
     * @param rho The momentum drag constant.
     * @param momentumType Momentum type.
     * @return A linear decay SGD with momentum.
     */
    public static SGD getLinearDecaySGD(double learningRate, double rho, Momentum momentumType) {
        return new LinearDecaySGD(learningRate, rho, momentumType);
    }

    /**
     * Generates an SGD optimiser with a sqrt decaying learning rate initialised to learningRate.
     * <p>
     * The learning rate = initialLearningRate / sqrt(iteration).
     * @param learningRate The learning rate.
     * @return A sqrt decay SGD.
     */
    public static SGD getSqrtDecaySGD(double learningRate) {
        return new SqrtDecaySGD(learningRate);
    }

    /**
     * Generates an SGD optimiser with a sqrt decaying learning rate initialised to learningRate, with momentum.
     * <p>
     * The learning rate = initialLearningRate / sqrt(iteration).
     * @param learningRate The learning rate.
     * @param rho The momentum drag constant.
     * @param momentumType Momentum type.
     * @return A sqrt decay SGD with momentum.
     */
    public static SGD getSqrtDecaySGD(double learningRate, double rho, Momentum momentumType) {
        return new SqrtDecaySGD(learningRate, rho, momentumType);
    }
}

final class SimpleSGD extends SGD {
    SimpleSGD(double learningRate) {
        super(learningRate);
    }

    SimpleSGD(double learningRate, double rho, Momentum momentumType) {
        super(learningRate, rho, momentumType);
    }

    /**
     * for OLCUT.
     */
    private SimpleSGD() { }

    @Override
    public double learningRate() {
        return initialLearningRate;
    }

    @Override
    protected String sgdType() {
        return "Constant";
    }

    @Override
    public SimpleSGD copy() {
        return new SimpleSGD(initialLearningRate,rho,useMomentum);
    }
}

final class LinearDecaySGD extends SGD {
    LinearDecaySGD(double learningRate) {
        super(learningRate);
    }

    LinearDecaySGD(double learningRate, double rho, Momentum momentumType) {
        super(learningRate, rho, momentumType);
    }

    /**
     * for OLCUT.
     */
    private LinearDecaySGD() { }

    @Override
    public double learningRate() {
        return initialLearningRate / iteration;
    }

    @Override
    protected String sgdType() {
        return "LinearDecay";
    }

    @Override
    public LinearDecaySGD copy() {
        return new LinearDecaySGD(initialLearningRate,rho,useMomentum);
    }
}

final class SqrtDecaySGD extends SGD {
    SqrtDecaySGD(double learningRate) {
        super(learningRate);
    }

    SqrtDecaySGD(double learningRate, double rho, Momentum momentumType) {
        super(learningRate, rho, momentumType);
    }

    /**
     * For OLCUT.
     */
    private SqrtDecaySGD() { }

    @Override
    public double learningRate() {
        return initialLearningRate / Math.sqrt(iteration);
    }

    @Override
    protected String sgdType() {
        return "SqrtDecay";
    }

    @Override
    public SqrtDecaySGD copy() {
        return new SqrtDecaySGD(initialLearningRate,rho,useMomentum);
    }
}
