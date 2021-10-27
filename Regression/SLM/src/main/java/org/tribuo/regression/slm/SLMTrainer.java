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

package org.tribuo.regression.slm;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.SparseTrainer;
import org.tribuo.WeightedExamples;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.regression.Regressor;
import org.tribuo.util.Util;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
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

    @Config(description="Maximum number of features to use.")
    protected int maxNumFeatures = -1;

    @Config(description="Normalize the data first.")
    protected boolean normalize;

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

    protected RealVector newWeights(SLMState state) {
        RealVector result = SLMTrainer.ordinaryLeastSquares(state.xpi,state.y);

        if (result == null) {
            return null;
        } else {
            return state.unpack(result);
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
        double[][] outputs = new double[numOutputs][numExamples];
        SparseVector[] inputs = new SparseVector[numExamples];
        int n = 0;
        for (Example<Regressor> e : examples) {
            inputs[n] = SparseVector.createSparseVector(e,featureIDMap,!normalize);
            double curWeight = Math.sqrt(e.getWeight());
            inputs[n].scaleInPlace(curWeight); //rescale features by example weight
            for (Regressor.DimensionTuple r : e.getOutput()) {
                int id = outputInfo.getID(r);
                outputs[id][n] = r.getValue() * curWeight; //rescale output by example weight
            }
            n++;
        }

        // Extract featureMatrix from the sparse vectors
        RealMatrix featureMatrix = new Array2DRowRealMatrix(numExamples, numFeatures);
        double[] denseFeatures = new double[numFeatures];
        for (int i = 0; i < inputs.length; i++) {
            Arrays.fill(denseFeatures,0.0);
            for (VectorTuple vec : inputs[i]) {
                denseFeatures[vec.index] = vec.value;
            }
            featureMatrix.setRow(i, denseFeatures);
        }

        double[] featureMeans = new double[numFeatures];
        double[] featureVariances = new double[numFeatures];
        double[] outputMeans = new double[numOutputs];
        double[] outputVariances = new double[numOutputs];
        if (normalize) {
            for (int i = 0; i < numFeatures; ++i) {
                double[] featV = featureMatrix.getColumn(i);
                featureMeans[i] = Util.mean(featV);

                for (int j=0; j < featV.length; ++j) {
                    featV[j] -= featureMeans[i];
                }

                RealVector xp = new ArrayRealVector(featV);
                featureVariances[i] = xp.getNorm();
                featureMatrix.setColumnVector(i,xp.mapDivideToSelf(featureVariances[i]));
            }

            for (int i = 0; i < numOutputs; i++) {
                outputMeans[i] = Util.mean(outputs[i]);
                // Remove mean and aggregate variance
                double sum = 0.0;
                for (int j = 0; j < numExamples; j++) {
                    outputs[i][j] -= outputMeans[i];
                    sum += outputs[i][j] * outputs[i][j];
                }
                outputVariances[i] = Math.sqrt(sum);
                // Remove variance
                for (int j = 0; j < numExamples; j++) {
                    outputs[i][j] /= outputVariances[i];
                }
            }
        } else {
            Arrays.fill(featureMeans,0.0);
            Arrays.fill(featureVariances,1.0);
            Arrays.fill(outputMeans,0.0);
            Arrays.fill(outputVariances,1.0);
        }

        // Construct the output matrix from the double[][] after scaling
        RealMatrix outputMatrix = new Array2DRowRealMatrix(outputs);

        // Array example is useful to compute a submatrix
        int[] exampleRows = new int[numExamples];
        for (int i = 0; i < numExamples; ++i) {
            exampleRows[i] = i;
        }

        RealVector one = new ArrayRealVector(numExamples,1.0);

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
            SLMState state = new SLMState(featureMatrix,outputMatrix.getRowVector(id),featureIDMap,normalize);
            modelWeights[id] = trainSingleDimension(state,exampleRows,numToSelect,one);
        }

        ModelProvenance provenance = new ModelProvenance(SparseLinearModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        return new SparseLinearModel("slm-model", dimensionNames, provenance, featureIDMap, outputInfo, modelWeights,
                DenseVector.createDenseVector(featureMeans), DenseVector.createDenseVector(featureVariances),
                outputMeans, outputVariances, !normalize);
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
     * @param exampleRows An array with the row indices in.
     * @param numToSelect The number of features to select.
     * @param one A RealVector of ones.
     * @return The sparse vector representing the learned feature weights.
     */
    private SparseVector trainSingleDimension(SLMState state, int[] exampleRows, int numToSelect, RealVector one) {
        int iter = 0;
        while (state.active.size() < numToSelect) {
            // Compute the residual
            state.r = state.y.subtract(state.X.operate(state.beta));

            logger.info("At iteration " + iter + " Average residual " + state.r.dotProduct(one) / state.numExamples);
            iter++;
            // Compute the correlation
            state.corr = state.X.transpose().operate(state.r);

            // Identify most correlated feature
            double max = -1;
            int feature = -1;
            for (int i = 0; i < state.numFeatures; ++i) {
                if (!state.activeSet.contains(i)) {
                    double absCorr = Math.abs(state.corr.getEntry(i));

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
            int[] activeFeatures = Util.toPrimitiveInt(state.active);
            state.xpi = state.X.getSubMatrix(exampleRows, activeFeatures);

            if (state.active.size() == (numToSelect - 1)) {
                state.last = true;
            }

            RealVector betapi = newWeights(state);

            if (betapi == null) {
                // Matrix was not invertible
                logger.log(Level.INFO, "Stopping at feature " + state.active.size() + " matrix was no longer invertible.");
                break;
            }

            state.beta = betapi;
        }

        Map<Integer, Double> parameters = new HashMap<>();

        for (int i = 0; i < state.numFeatures; ++i) {
            if (state.beta.getEntry(i) != 0) {
                parameters.put(i, state.beta.getEntry(i));
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
    static RealVector ordinaryLeastSquares(RealMatrix M, RealVector target) {
        RealMatrix inv;
        try {
            inv = new LUDecomposition(M.transpose().multiply(M)).getSolver().getInverse();
        } catch (SingularMatrixException s) {
            // Matrix is not invertible, there is nothing we can do
            // We will let the caller decide what to do
            return null;
        }

        return inv.multiply(M.transpose()).operate(target);
    }

    /**
     * Sums inverted matrix.
     * @param matrix The Matrix to operate on.
     * @return The sum of the inverted matrix.
     */
    static double sumInverted(RealMatrix matrix) {
        // Why are we not trying to catch the potential exception?
        // Because in the context of LARS, if we call this method, we know the matrix is invertible
        RealMatrix inv = new LUDecomposition(matrix.transpose().multiply(matrix)).getSolver().getInverse();

        RealVector one = new ArrayRealVector(matrix.getColumnDimension(),1.0);

        return one.dotProduct(inv.operate(one));
    }

    /**
     * Inverts the matrix, takes the dot product and scales it by the supplied value.
     * @param M The matrix to invert.
     * @param AA The value to scale by.
     * @return The vector of feature values.
     */
    static RealVector getwa(RealMatrix M, double AA) {
        RealMatrix inv = new LUDecomposition(M.transpose().multiply(M)).getSolver().getInverse();
        RealVector one = new ArrayRealVector(M.getColumnDimension(),1.0);

        return inv.operate(one).mapMultiply(AA);
    }

    /**
     * Calculates (M . v) . D^T
     * Used in LARS.
     * @param D A matrix.
     * @param M A matrix.
     * @param v A vector.
     * @return (M . v) . D^T
     */
    static RealVector getA(RealMatrix D, RealMatrix M, RealVector v) {
        RealVector u = M.operate(v);
        return D.transpose().operate(u);
    }

    static class SLMState {
        protected final int numExamples;
        protected final int numFeatures;
        protected final boolean normalize;
        protected final ImmutableFeatureMap featureIDMap;

        protected final Set<Integer> activeSet;
        protected final List<Integer> active;

        protected final RealMatrix X;
        protected final RealVector y;

        protected RealMatrix xpi;
        protected RealVector r;
        protected RealVector beta;

        protected double C;
        protected RealVector corr;

        protected Boolean last = false;

        public SLMState(RealMatrix features, RealVector outputs, ImmutableFeatureMap featureIDMap, boolean normalize) {
            this.numExamples = features.getRowDimension();
            this.numFeatures = features.getColumnDimension();
            this.featureIDMap = featureIDMap;
            this.normalize = normalize;
            this.active = new ArrayList<>();
            this.activeSet = new HashSet<>();
            this.beta = new ArrayRealVector(numFeatures);
            this.X = features;
            this.y = outputs;
        }

        /**
         * Unpacks the active set into a dense vector using the values in values
         * @param values The values.
         * @return A dense vector representing the values at the active set indices.
         */
        public RealVector unpack(RealVector values) {
            RealVector u = new ArrayRealVector(numFeatures);

            for (int i = 0; i < active.size(); ++i) {
                u.setEntry(active.get(i), values.getEntry(i));
            }

            return u;
        }
    }
}
