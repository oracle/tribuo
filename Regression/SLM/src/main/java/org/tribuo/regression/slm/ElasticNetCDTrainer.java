/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.regression.slm;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.SparseModel;
import org.tribuo.SparseTrainer;
import org.tribuo.Trainer;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.Regressor.DimensionTuple;
import org.tribuo.util.Util;

import java.time.OffsetDateTime;
import java.util.Arrays;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.tribuo.math.la.VectorTuple.DELTA;

/**
 * An ElasticNet trainer that uses co-ordinate descent. Modelled after scikit-learn's sparse matrix implementation.
 * Each output dimension is trained independently.
 * <p>
 * See:
 * <pre>
 * Friedman J, Hastie T, Tibshirani R.
 * "Regularization Paths for Generalized Linear Models via Coordinate Descent"
 * Journal of Statistical Software, 2010
 * </pre>
 */
public class ElasticNetCDTrainer implements SparseTrainer<Regressor> {

    private static final Logger logger = Logger.getLogger(ElasticNetCDTrainer.class.getName());

    @Config(mandatory = true,description="Overall regularisation penalty.")
    private double alpha;

    @Config(mandatory = true,description="Ratio of l1 to l2 parameters.")
    private double l1Ratio;

    @Config(description="Tolerance on the error.")
    private double tolerance = 1e-4;

    @Config(description="Maximium number of iterations to run.")
    private int maxIterations = 500;

    @Config(description="Randomises the order in which the features are probed.")
    private boolean randomise = false;

    @Config(description="The seed for the RNG.")
    private long seed = Trainer.DEFAULT_SEED;

    private SplittableRandom rng;

    private int trainInvocationCounter;

    /**
     * For olcut.
     */
    private ElasticNetCDTrainer() { }

    /**
     * Constructs an elastic net trainer using the supplied parameters, with a tolerance of 1e-4, max iterations of 500, randomising the feature choice and using {@link Trainer#DEFAULT_SEED} as the RNG seed.
     * @param alpha The regularisation strength.
     * @param l1Ratio The ratio between the l1 and l2 penalties.
     */
    public ElasticNetCDTrainer(double alpha, double l1Ratio) {
        this(alpha,l1Ratio,1e-4,500,false,Trainer.DEFAULT_SEED);
    }

    /**
     * Constructs an elastic net trainer using the supplied parameters, with a tolerance of 1e-4, max iterations of 500, and randomising the feature choice.
     * @param alpha The regularisation strength.
     * @param l1Ratio The ratio between the l1 and l2 penalties.
     * @param seed The RNG seed.
     */
    public ElasticNetCDTrainer(double alpha, double l1Ratio, long seed) {
        this(alpha,l1Ratio,1e-4,500,true,seed);
    }

    /**
     * Constructs an elastic net trainer using the supplied parameters.
     * @param alpha The regularisation strength.
     * @param l1Ratio The ratio between the l1 and l2 penalties.
     * @param tolerance The convergence tolerance.
     * @param maxIterations The maximum number of iterations.
     * @param randomise Randomise the feature choice order.
     * @param seed The RNG seed.
     */
    public ElasticNetCDTrainer(double alpha, double l1Ratio, double tolerance, int maxIterations, boolean randomise, long seed) {
        this.alpha = alpha;
        this.l1Ratio = l1Ratio;
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
        this.randomise = randomise;
        this.seed = seed;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public synchronized void postConfig() {
        if ((l1Ratio < DELTA) || (l1Ratio > 1.0 + DELTA)) {
            throw new PropertyException("l1Ratio","L1 Ratio must be between 0 and 1. Found value " + l1Ratio);
        }
        this.rng = new SplittableRandom(seed);
    }

    @Override
    public SparseModel<Regressor> train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance) {
        return train(examples, runProvenance, INCREMENT_INVOCATION_COUNT);
    }

    @Override
    public SparseModel<Regressor> train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        // Creates a new RNG, adds one to the invocation count, generates provenance.
        TrainerProvenance trainerProvenance;
        SplittableRandom localRNG;
        synchronized(this) {
            if(invocationCount != INCREMENT_INVOCATION_COUNT) {
                setInvocationCount(invocationCount);
            }
            localRNG = rng.split();
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableFeatureMap featureIDMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<Regressor> outputInfo = examples.getOutputIDInfo();
        int numFeatures = featureIDMap.size();
        int numOutputs = outputInfo.size();
        int numExamples = examples.size();
        SparseVector[] columns = SparseVector.transpose(examples,featureIDMap);
        String[] dimensionNames = new String[numOutputs];
        DenseVector[] regressionTargets = new DenseVector[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            dimensionNames[i] = outputInfo.getOutput(i).getNames()[0];
            regressionTargets[i] = new DenseVector(numExamples);
        }
        int i = 0;
        for (Example<Regressor> e : examples) {
            for (DimensionTuple d : e.getOutput()) {
                regressionTargets[outputInfo.getID(d)].set(i, d.getValue());
            }
            i++;
        }
        double l1Penalty = alpha * l1Ratio * numExamples;
        double l2Penalty = alpha * (1.0 - l1Ratio) * numExamples;

        double[] featureMeans = calculateMeans(columns);
        double[] featureVariances = new double[columns.length];
        Arrays.fill(featureVariances,1.0);
        boolean center = false;
        for (i = 0; i < numFeatures; i++) {
            if (Math.abs(featureMeans[i]) > DELTA) {
                center = true;
                break;
            }
        }
        double[] columnNorms = new double[numFeatures];
        int[] featureIndices = new int[numFeatures];

        for (i = 0; i < numFeatures; i++) {
            featureIndices[i] = i;
            double variance = 0.0;
            for (VectorTuple v : columns[i]) {
                variance += (v.value - featureMeans[i]) * (v.value - featureMeans[i]);
            }
            columnNorms[i] = variance + (numExamples - columns[i].numActiveElements()) * featureMeans[i] * featureMeans[i];
        }

        ElasticNetState elState = new ElasticNetState(columns,featureIndices,featureMeans,columnNorms,l1Penalty,l2Penalty,center);

        SparseVector[] outputWeights = new SparseVector[numOutputs];
        double[] outputMeans = new double[numOutputs];
        for (int j = 0; j < dimensionNames.length; j++) {
            outputWeights[j] = trainSingleDimension(regressionTargets[j],elState,localRNG.split());
            outputMeans[j] = regressionTargets[j].sum() / numExamples;
        }
        double[] outputVariances = new double[numOutputs];//calculateVariances(regressionTargets,outputMeans);
        Arrays.fill(outputVariances,1.0);

        ModelProvenance provenance = new ModelProvenance(SparseLinearModel.class.getName(), OffsetDateTime.now(),examples.getProvenance(),trainerProvenance,runProvenance);
        return new SparseLinearModel("elastic-net-model", dimensionNames, provenance, featureIDMap, outputInfo,
                outputWeights, DenseVector.createDenseVector(featureMeans), DenseVector.createDenseVector(featureVariances),
                outputMeans, outputVariances, false);
    }

    private SparseVector trainSingleDimension(DenseVector regressionTargets, ElasticNetState state, SplittableRandom localRNG) {
        int numFeatures = state.numFeatures;
        int numExamples = state.numExamples;
        DenseVector residuals = regressionTargets.copy();
        DenseVector weights = new DenseVector(numFeatures);
        double targetTwoNorm = regressionTargets.twoNorm();
        double newTolerance = tolerance * targetTwoNorm * targetTwoNorm;

        double[] xTransposeR = new double[numFeatures];
        double[] xTransposeAlpha = new double[numFeatures];

        for (int i = 0; i < maxIterations; i++) {
            double maxWeight = 0.0;
            double maxUpdate = 0.0;

            // If randomly selecting the features, permute the indices
            if (randomise) {
                Util.randpermInPlace(state.featureIndices,localRNG);
            }

            // Iterate through the features
            for (int j = 0; j < numFeatures; j++) {
                int feature = state.featureIndices[j];

                if (Math.abs(state.columnNorms[feature]) < DELTA) {
                    continue;
                }

                double oldWeight = weights.get(feature);

                // Update residual
                if (oldWeight != 0.0) {
                    for (VectorTuple v : state.columns[feature]) {
                        residuals.set(v.index, residuals.get(v.index) + (v.value * oldWeight));
                    }
                    if (state.center) {
                        for (int k = 0; k < numExamples; k++) {
                            residuals.set(k, residuals.get(k) - (state.featureMeans[feature] * oldWeight));
                        }
                    }
                }

                // Update the weights in the required direction
                double curDot = residuals.dot(state.columns[feature]);
                if (state.center) {
                    curDot -= residuals.sum() * state.featureMeans[feature];
                }
                double newWeight = Math.signum(curDot) * Math.max(Math.abs(curDot) - state.l1Penalty, 0) / (state.columnNorms[feature] + state.l2Penalty);
                weights.set(feature,newWeight);

                // Update residual after step
                if (newWeight != 0.0) {
                    for (VectorTuple v : state.columns[feature]) {
                        residuals.set(v.index, residuals.get(v.index) - (v.value * newWeight));
                    }
                    if (state.center) {
                        for (int k = 0; k < numExamples; k++) {
                            residuals.set(k, residuals.get(k) + (state.featureMeans[feature] * newWeight));
                        }
                    }
                }

                double curUpdate = Math.abs(newWeight - oldWeight);

                if (curUpdate > maxUpdate) {
                    maxUpdate = curUpdate;
                }

                double absNewWeight = Math.abs(newWeight);
                if (absNewWeight > maxWeight) {
                    maxWeight = absNewWeight;
                }
            }

            //logger.log(Level.INFO, "Iteration " + i + ", average residual = " + residuals.sum()/numExamples);

            // Check the termination condition
            if ((maxWeight < DELTA) || (maxUpdate / maxWeight < tolerance) || (i == (maxIterations-1))) {
                double residualSum = residuals.sum();

                double maxAbsXTA = 0.0;
                for (int j = 0; j < numFeatures; j++) {
                    xTransposeR[j] = residuals.dot(state.columns[j]);

                    if (state.center) {
                        xTransposeR[j] -= state.featureMeans[j] * residualSum;
                    }

                    xTransposeAlpha[j] = xTransposeR[j] - state.l2Penalty * weights.get(j);

                    double curAbs = Math.abs(xTransposeAlpha[j]);
                    if (curAbs > maxAbsXTA) {
                        maxAbsXTA = curAbs;
                    }
                }

                double residualTwoNorm = residuals.twoNorm();
                residualTwoNorm *= residualTwoNorm;

                double weightsTwoNorm = weights.twoNorm();
                weightsTwoNorm *= weightsTwoNorm;

                double weightsOneNorm = weights.oneNorm();

                double scalingFactor, dualityGap;
                if (maxAbsXTA > state.l1Penalty) {
                    scalingFactor = state.l1Penalty / maxAbsXTA;
                    double alphaNorm = residualTwoNorm * scalingFactor * scalingFactor;
                    dualityGap = 0.5 * (residualTwoNorm + alphaNorm);
                } else {
                    scalingFactor = 1.0;
                    dualityGap = residualTwoNorm;
                }

                dualityGap += state.l1Penalty * weightsOneNorm - scalingFactor * residuals.dot(regressionTargets);
                dualityGap += 0.5 * state.l2Penalty * (1 + (scalingFactor * scalingFactor)) * weightsTwoNorm;

                if (dualityGap < newTolerance) {
                    // All done, stop iterating.
                    logger.log(Level.INFO,"Iteration: " + i + ", duality gap = " + dualityGap + ", tolerance = " + newTolerance);
                    break;
                }
            }
        }


        return weights.sparsify();
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

        rng = new SplittableRandom(seed);
        for (trainInvocationCounter = 0; trainInvocationCounter < invocationCount; trainInvocationCounter++){
            SplittableRandom localRNG = rng.split();
        }
    }

    @Override
    public String toString() {
        return "ElasticNetCDTrainer(alpha="+alpha+",l1Ratio="+l1Ratio+"" +
                ",tolerance="+tolerance+",maxIterations="+maxIterations +
                ",randomise="+randomise+",seed="+seed+")";
    }

    private static double[] calculateMeans(SGDVector[] columns) {
        double[] means = new double[columns.length];

        for (int i = 0; i < means.length; i++) {
            means[i] = columns[i].sum() / columns[i].size();
        }

        return means;
    }

    private static double[] calculateVariances(SGDVector[] columns, double[] means) {
        double[] variances = new double[columns.length];

        for (int i = 0; i < variances.length; i++) {
            variances[i] = columns[i].variance(means[i]);
        }

        return variances;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }

    /**
     * Carrier type for the immutable elastic net state.
     */
    private static class ElasticNetState {
        final SparseVector[] columns;
        final int numFeatures;
        final int numExamples;
        final int[] featureIndices;
        final double[] featureMeans;
        final double[] columnNorms;
        final double l1Penalty;
        final double l2Penalty;
        final boolean center;

        public ElasticNetState(SparseVector[] columns, int[] featureIndices, double[] featureMeans, double[] columnNorms, double l1Penalty, double l2Penalty, boolean center) {
            this.columns = columns;
            this.numFeatures = columns.length;
            this.numExamples = columns[0].size();
            this.featureIndices = featureIndices;
            this.featureMeans = featureMeans;
            this.columnNorms = columnNorms;
            this.l1Penalty = l1Penalty;
            this.l2Penalty = l2Penalty;
            this.center = center;
        }
    }
}
