/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.regression.sgd.linear;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.optimisers.LBFGS;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.sgd.RegressionObjective;
import org.tribuo.regression.sgd.objectives.SquaredLoss;

import java.time.OffsetDateTime;
import java.util.Map;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.logging.Logger;

/**
 * A trainer for a linear regression using L-BFGS.
 * <p>
 * See:
 * <pre>
 * Nocedal, J. and Wright, S.
 * "Numerical Optimization (2nd Edition)"
 * Springer, 2006.
 * </pre>
 */
public final class LinearTrainer implements Trainer<Regressor> {
    private static final Logger logger = Logger.getLogger(LinearTrainer.class.getName());

    @Config(description = "The regression objective function to use.")
    private RegressionObjective objective = new SquaredLoss();

    @Config(description="The number of optimization iterations.")
    private int maxIterations = 100;

    @Config(description = "Use L2 regularization.")
    private boolean l2Penalty = false;

    @Config(description = "Tolerance stopping criterion.")
    private double tolerance = 1e-4;

    @Config(description = "Gradient tolerance stopping criterion.")
    private double gradientTolerance = 1e-4;

    @Config(description = "Regularization strength.")
    private double regularisationStrength = 1.0;

    @Config(description = "Memory size for L-BFGS")
    private int memorySize = 10;

    private int trainInvocationCounter = 0;

    /**
     * Constructs a trainer for a linear model using L-BFGS.
     *
     * @param objective       The objective function to optimise.
     */
    public LinearTrainer(RegressionObjective objective, int maxIterations, boolean l2Penalty, double tolerance, double gradientTolerance, double regularisationStrength) {
        this.objective = objective;
        this.maxIterations = maxIterations;
        this.l2Penalty = l2Penalty;
        this.tolerance = tolerance;
        this.gradientTolerance = gradientTolerance;
        this.regularisationStrength = regularisationStrength;
        postConfig();
    }
    /**
     * For OLCUT.
     */
    private LinearTrainer() {
        super();
    }

    @Override
    public void postConfig() {}

    @Override
    public String toString() {
        return "LinearTrainer(" +
                "objective=" + objective +
                ", maxIterations=" + maxIterations +
                ", l2Penalty=" + l2Penalty +
                ", tolerance=" + tolerance +
                ", gradientTolerance=" + gradientTolerance +
                ", regularisationStrength=" + regularisationStrength +
                ", memorySize=" + memorySize +
                ')';
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public synchronized void setInvocationCount(int invocationCount){
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }
        trainInvocationCounter = invocationCount;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }

    @Override
    public Model<Regressor> train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        // Adds one to the invocation count, creates provenance.
        TrainerProvenance trainerProvenance;
        synchronized(this) {
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }

        ImmutableOutputInfo<Regressor> outputIDInfo = examples.getOutputIDInfo();
        ImmutableFeatureMap featureIDMap = examples.getFeatureIDMap();
        int featureSpaceSize = featureIDMap.size();
        int numOutputs = outputIDInfo.size();
        DenseVector[] sgdFeatures = new DenseVector[examples.size()];
        DenseVector[] sgdTargets = new DenseVector[examples.size()];
        double[] weights = new double[examples.size()];
        double weightSum = 0.0;
        int n = 0;
        long featureSize = 0;
        for (Example<Regressor> example : examples) {
            double w = example.getWeight();
            weights[n] = w;
            weightSum += w;
            sgdFeatures[n] = DenseVector.createDenseVector(example, featureIDMap, true);
            double[] regressorsBuffer = new double[outputIDInfo.size()];
            for (Regressor.DimensionTuple r : example.getOutput()) {
                int id = outputIDInfo.getID(r);
                regressorsBuffer[id] = r.getValue();
            }
            sgdTargets[n] = DenseVector.createDenseVector(regressorsBuffer);
            featureSize += sgdFeatures[n].numActiveElements();
            n++;
        }
        // normalize weights so they sum to 1.
        for (int i = 0; i < weights.length; i++) {
            weights[i] = weights[i] / weightSum;
        }
        DenseVector weightVec = DenseVector.createDenseVector(weights);
        DenseMatrix dataMatrix = DenseMatrix.createDenseMatrix(sgdFeatures);
        DenseMatrix outputMatrix = DenseMatrix.createDenseMatrix(sgdTargets);
        logger.info(String.format("Training SGD model with %d examples", n));
        logger.fine("Mean number of active features = " + featureSize / (double)n);
        logger.info("Outputs - " + outputIDInfo.toReadableString());

        // Includes bias
        LinearParameters parameters = new LinearParameters(featureSpaceSize+1,numOutputs);

        double l2RegStrength = 1.0 / (this.regularisationStrength * weightSum);

        logger.info(String.format("Training linear model with %d examples and %d features", featureSpaceSize, numOutputs));
        LBFGS lbfgs = new LBFGS(memorySize, maxIterations, tolerance, gradientTolerance);

        Function<Tensor[], LBFGS.GradAndLoss> obj = (Tensor[] params) -> {
            DenseMatrix pred = dataMatrix.matrixMultiply(((Matrix) params[0]), false, true);
            var p = objective.batchLossAndGradient(outputMatrix, pred);
            p.gradient().rowScaleInPlace(weightVec);
            Matrix gradient = p.gradient().matrixMultiply(dataMatrix, true, false);
            double loss = 0.0;
            for (int i = 0; i < weights.length; i++) {
                loss += p.loss()[i] * weights[i];
            }
            return new LBFGS.GradAndLoss(new Tensor[]{gradient}, loss);
        };

        ToDoubleFunction<Tensor[]> lossFunc = (Tensor[] params) -> {
            DenseMatrix pred = dataMatrix.matrixMultiply(((Matrix) params[0]), false, true);
            double[] lossArr = objective.batchLoss(outputMatrix, pred);
            double loss = 0.0;
            for (int i = 0; i < weights.length; i++) {
                loss += lossArr[i] * weights[i];
            }
            return loss;
        };

        lbfgs.optimize(parameters, obj, lossFunc);

        ModelProvenance provenance = new ModelProvenance(LinearTrainer.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        String[] dimensionNames = new String[outputIDInfo.size()];
        for (Regressor r : outputIDInfo.getDomain()) {
            int id = outputIDInfo.getID(r);
            dimensionNames[id] = r.getNames()[0];
        }
        return new LinearSGDModel("linear-model", dimensionNames, provenance, featureIDMap, outputIDInfo, parameters);
    }
}
