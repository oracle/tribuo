/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.SparseTrainer;
import org.tribuo.WeightedExamples;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.regression.Regressor;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A trainer for a sparse linear regression model.
 * Uses sequential forward selection to construct the model. Optionally can
 * normalize the data first. Each output dimension is trained independently
 * with no shared regularization.
 */
public class SLMTrainer implements SparseTrainer<Regressor>, WeightedExamples {
    private static final Logger logger = Logger.getLogger(SLMTrainer.class.getName());

    /**
     * The maximum number of features to select.
     */
    @Config(description="Maximum number of features to use.")
    protected int maxNumFeatures = -1;

    /**
     * Should the data be centred first? In most cases this should be true.
     */
    @Config(description="Normalize the data first.")
    protected boolean normalize;

    /**
     * The number of times {@link #train} has been called on this object.
     */
    protected int trainInvocationCounter = 0;

    /**
     * Constructs a trainer for a sparse linear model using sequential forward selection.
     *
     * @param normalize Normalizes the data first (i.e., removes the bias term).
     * @param maxNumFeatures The maximum number of features to select. Supply -1 to select all features.
     */
    public SLMTrainer(boolean normalize, int maxNumFeatures) {
        this.normalize = normalize;
        this.maxNumFeatures = maxNumFeatures;
    }

    /**
     * Constructs a trainer for a sparse linear model using sequential forward selection.
     * <p>
     * Selects all the features.
     *
     * @param normalize Normalizes the data first (i.e., removes the bias term).
     */
    public SLMTrainer(boolean normalize) {
        this(normalize,-1);
    }

    /**
     * For OLCUT.
     */
    protected SLMTrainer() {}

    /**
     * Computes the new feature weights.
     * <p>
     * In this version it returns the ordinary least squares solution for the current state.
     * @param state The SLM state to operate on.
     * @return The new feature weights.
     */
    protected DenseVector newWeights(SLMState state) {
        Pair<DenseVector,DenseMatrix> result = SLMTrainer.ordinaryLeastSquares(state.xpi,state.y);

        if (result == null) {
            return null;
        } else {
            return state.unpack(result.getA());
        }
    }

    /**
     * Trains a sparse linear model.
     * @param examples The data set containing the examples.
     * @return A trained sparse linear model.
     */
    @Override
    public SparseLinearModel train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance) {
        return train(examples, runProvenance, INCREMENT_INVOCATION_COUNT);
    }


    /**
     * Trains a sparse linear model.
     * @param examples The data set containing the examples.
     * @param invocationCount The state of the RNG the trainer should be set to before training
     * @return A trained sparse linear model.
     */
    @Override
    public SparseLinearModel train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }

        TrainerProvenance trainerProvenance;
        synchronized(this) {
            if(invocationCount != INCREMENT_INVOCATION_COUNT) {
                setInvocationCount(invocationCount);
            }
            trainerProvenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableOutputInfo<Regressor> outputInfo = examples.getOutputIDInfo();
        ImmutableFeatureMap featureIDMap = examples.getFeatureIDMap();
        Set<Regressor> domain = outputInfo.getDomain();
        int numOutputs = outputInfo.size();
        int numExamples = examples.size();
        int numFeatures = normalize ? featureIDMap.size() : featureIDMap.size() + 1; //include bias
        DenseMatrix outputMatrix = new DenseMatrix(numOutputs,numExamples);
        SparseVector[] inputs = new SparseVector[numExamples];
        int n = 0;
        for (Example<Regressor> e : examples) {
            inputs[n] = SparseVector.createSparseVector(e,featureIDMap,!normalize);
            double curWeight = Math.sqrt(e.getWeight());
            inputs[n].scaleInPlace(curWeight); //rescale features by example weight
            for (Regressor.DimensionTuple r : e.getOutput()) {
                int id = outputInfo.getID(r);
                outputMatrix.set(id,n,r.getValue() * curWeight); //rescale output by example weight
            }
            n++;
        }

        // Extract featureMatrix from the sparse vectors
        DenseMatrix featureMatrix = DenseMatrix.createDenseMatrix(inputs);

        double[] featureMeans = new double[numFeatures];
        double[] featureNorms = new double[numFeatures];
        double[] outputMeans = new double[numOutputs];
        double[] outputNorms = new double[numOutputs];
        if (normalize) {
            for (int i = 0; i < numFeatures; ++i) {
                DenseVector col = featureMatrix.getColumn(i);
                double colMean = col.meanVariance().getMean();
                double colNorm = Math.sqrt(col.reduce(0.0, a -> a - colMean, (a,b) -> b + a*a));
                col.foreachInPlace(a -> (a - colMean) / colNorm);
                featureMatrix.setColumn(i,col);
                featureMeans[i] = colMean;
                featureNorms[i] = colNorm;
            }

            for (int i = 0; i < numOutputs; i++) {
                DenseVector row = outputMatrix.getRow(i);
                double rowMean = row.meanVariance().getMean();
                double rowNorm = Math.sqrt(row.reduce(0.0, a -> a - rowMean, (a,b) -> b + a*a));
                row.foreachInPlace(a -> (a - rowMean) / rowNorm);
                outputMeans[i] = rowMean;
                outputNorms[i] = rowNorm;
            }
        } else {
            Arrays.fill(featureMeans,0.0);
            Arrays.fill(featureNorms,1.0);
            Arrays.fill(outputMeans,0.0);
            Arrays.fill(outputNorms,1.0);
        }

        int numToSelect;
        if ((maxNumFeatures < 1) || (maxNumFeatures > featureIDMap.size())) {
            numToSelect = featureIDMap.size();
        } else {
            numToSelect = maxNumFeatures;
        }

        String[] dimensionNames = new String[numOutputs];
        SparseVector[] modelWeights = new SparseVector[numOutputs];
        for (Regressor r : domain) {
            int id = outputInfo.getID(r);
            dimensionNames[id] = r.getNames()[0];
            SLMState state = new SLMState(featureMatrix,outputMatrix.getRow(id),featureIDMap,normalize);
            modelWeights[id] = trainSingleDimension(state,numToSelect);
        }

        ModelProvenance provenance = new ModelProvenance(SparseLinearModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        return new SparseLinearModel("slm-model", dimensionNames, provenance, featureIDMap, outputInfo, modelWeights,
                DenseVector.createDenseVector(featureMeans), DenseVector.createDenseVector(featureNorms),
                outputMeans, outputNorms, !normalize);
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public void setInvocationCount(int invocationCount) {
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }

        this.trainInvocationCounter = invocationCount;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }

    @Override
    public String toString() {
        return "SFSTrainer(normalize="+normalize+",maxNumFeatures="+maxNumFeatures+")";
    }

    /**
     * Trains a single dimension.
     * @param state The state object to use.
     * @param numToSelect The number of features to select.
     * @return The sparse vector representing the learned feature weights.
     */
    private SparseVector trainSingleDimension(SLMState state, int numToSelect) {
        int iter = 0;
        while (state.active.size() < numToSelect) {
            // Compute the residual
            state.r = state.y.subtract(state.X.leftMultiply(state.beta));

            logger.info("At iteration " + iter + " Average residual " + state.r.sum() / state.numExamples);
            iter++;
            // Compute the correlation
            state.corr = state.X.rightMultiply(state.r);

            // Identify most correlated feature
            double max = -1;
            int feature = -1;
            for (int i = 0; i < state.numFeatures; ++i) {
                if (!state.activeSet.contains(i)) {
                    double absCorr = Math.abs(state.corr.get(i));

                    if (absCorr > max) {
                        max = absCorr;
                        feature = i;
                    }
                }
            }

            state.C = max;

            state.active.add(feature);
            state.activeSet.add(feature);

            if (!state.normalize && (feature == state.numFeatures-1)) {
                logger.info("Bias selected");
            } else {
                logger.info("Feature selected: " + state.featureIDMap.get(feature).getName() + " (pos=" + feature + ")");
            }

            // Compute the active matrix
            state.xpi = state.X.selectColumns(state.active);

            if (state.active.size() == (numToSelect - 1)) {
                state.last = true;
            }

            DenseVector betapi = newWeights(state);

            if (betapi == null) {
                // Matrix was not invertible
                logger.log(Level.INFO, "Stopping at feature " + state.active.size() + " matrix was no longer invertible.");
                break;
            }

            state.beta = betapi;
        }

        Map<Integer, Double> parameters = new HashMap<>();

        for (int i = 0; i < state.numFeatures; ++i) {
            if (state.beta.get(i) != 0) {
                parameters.put(i, state.beta.get(i));
            }
        }

        return SparseVector.createSparseVector(state.numFeatures, parameters);
    }

    /**
     * Minimize ordinary least squares.
     *
     * Returns null if the matrix is not invertible.
     * @param M The matrix of features.
     * @param target The vector of target values.
     * @return The OLS solution for the supplied features.
     */
    static Pair<DenseVector,DenseMatrix> ordinaryLeastSquares(DenseMatrix M, DenseVector target) {
        Optional<DenseMatrix.LUFactorization> lu = M.matrixMultiply(M,true,false).luFactorization();
        if (lu.isPresent()) {
            DenseMatrix inv = (DenseMatrix) lu.get().inverse();
            return new Pair<>(inv.matrixMultiply(M,false,true).leftMultiply(target),inv);
        } else {
            // Matrix is not invertible, there is nothing we can do
            // We will let the caller decide what to do
            return null;
        }
    }

    static DenseVector getWA(DenseMatrix inv, double AA) {
        DenseVector ones = new DenseVector(inv.getDimension2Size(),1.0);
        DenseVector output = inv.rightMultiply(ones);
        output.scaleInPlace(AA);
        return output;
    }

    /**
     * Calculates (M . v) . D^T
     * Used in LARS.
     * @param D A matrix.
     * @param M A matrix.
     * @param v A vector.
     * @return (M . v) . D^T
     */
    static DenseVector getA(DenseMatrix D, DenseMatrix M, DenseVector v) {
        DenseVector u = M.leftMultiply(v);
        return D.rightMultiply(u);
    }

    static class SLMState {
        protected final int numExamples;
        protected final int numFeatures;
        protected final boolean normalize;
        protected final ImmutableFeatureMap featureIDMap;

        protected final Set<Integer> activeSet;
        protected final List<Integer> active;

        protected final DenseMatrix X;
        protected final DenseVector y;

        protected DenseMatrix xpi;
        protected DenseVector r;
        protected DenseVector beta;

        protected double C;
        protected DenseVector corr;

        protected boolean last = false;

        public SLMState(DenseMatrix features, DenseVector outputs, ImmutableFeatureMap featureIDMap, boolean normalize) {
            this.numExamples = features.getDimension1Size();
            this.numFeatures = features.getDimension2Size();
            this.featureIDMap = featureIDMap;
            this.normalize = normalize;
            this.active = new ArrayList<>(numFeatures);
            this.activeSet = new HashSet<>();
            this.beta = new DenseVector(numFeatures);
            this.X = features;
            this.y = outputs;
        }

        /**
         * Unpacks the active set into a dense vector using the values in values
         * @param values The values.
         * @return A dense vector representing the values at the active set indices.
         */
        public DenseVector unpack(DenseVector values) {
            DenseVector u = new DenseVector(numFeatures);

            for (int i = 0; i < active.size(); ++i) {
                u.set(active.get(i), values.get(i));
            }

            return u;
        }
    }
}
