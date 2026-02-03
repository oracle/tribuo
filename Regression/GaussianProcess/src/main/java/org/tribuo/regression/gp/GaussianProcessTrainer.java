/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.regression.gp;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.math.kernel.Kernel;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.regression.Regressor;
import org.tribuo.util.MeanVarianceAccumulator;

import java.time.OffsetDateTime;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

/**
 * Implements a Gaussian Process regression.
 * <p>
 * Note this implementation is not approximate and requires inverting the data matrix, so should only be
 * used for small numbers of examples.
 * <p>
 * See:
 * <pre>
 * Rasmussen C, Williams C.
 * "Gaussian Processes for Machine Learning"
 * MIT Press, 2006.
 * </pre>
 */
public final class GaussianProcessTrainer implements Trainer<Regressor> {

    /**
     * Kernel function.
     */
    @Config(mandatory = true, description = "Kernel function.")
    private Kernel kernel;

    /**
     * Noise level, must be non-negative.
     */
    @Config(mandatory = true, description = "Noise level, must be non-negative.")
    private double alpha;

    /**
     * Should the kernel hyperparameters be optimized?
     */
    @Config(mandatory = false, description = "Should the kernel hyperparameters be optimized?")
    private boolean optimizeKernel;

    private int trainInvocationCount = 0;

    /**
     * For OLCUT.
     */
    private GaussianProcessTrainer() {}

    /**
     * Constructs a GP trainer with the specified parameters.
     * @param kernel The kernel function.
     * @param alpha The noise level.
     * @param optimizeKernel Should the kernel hyperparameters be optimized?
     */
    public GaussianProcessTrainer(Kernel kernel, double alpha, boolean optimizeKernel) {
        this.kernel = kernel;
        this.alpha = alpha;
        this.optimizeKernel = optimizeKernel;
        postConfig();
    }

    /**
     * For OLCUT, should not be called by user code.
     */
    public void postConfig() {
        if (alpha < 0.0) {
            throw new PropertyException("","alpha","Invalid alpha value, must be non-negative but found " + alpha);
        }
    }

    @Override
    public GaussianProcessModel train(Dataset<Regressor> examples) {
        return train(examples, Collections.emptyMap());
    }

    @Override
    public GaussianProcessModel train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance) {
        return train(examples, runProvenance, INCREMENT_INVOCATION_COUNT);
    }

    @Override
    public GaussianProcessModel train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }

        TrainerProvenance trainerProvenance;
        synchronized(this) {
            if(invocationCount != INCREMENT_INVOCATION_COUNT) {
                setInvocationCount(invocationCount);
            }
            trainerProvenance = getProvenance();
            trainInvocationCount++;
        }
        ImmutableOutputInfo<Regressor> outputInfo = examples.getOutputIDInfo();
        ImmutableFeatureMap featureIDMap = examples.getFeatureIDMap();
        Set<Regressor> domain = outputInfo.getDomain();
        int numOutputs = outputInfo.size();
        int numExamples = examples.size();
        int numFeatures = featureIDMap.size();
        DenseMatrix outputMatrix = new DenseMatrix(numExamples,numOutputs);
        SGDVector[] inputs = new SGDVector[numExamples];
        MeanVarianceAccumulator[] means = new MeanVarianceAccumulator[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            means[i] = new MeanVarianceAccumulator();
        }
        double[] weights = new double[numExamples];
        int n = 0;
        for (Example<Regressor> e : examples) {
            if (e.size() == numFeatures) {
                inputs[n] = DenseVector.createDenseVector(e, featureIDMap, false);
            } else {
                inputs[n] = SparseVector.createSparseVector(e, featureIDMap, false);
            }
            double curWeight = Math.sqrt(e.getWeight());
            weights[n] = curWeight;
            for (Regressor.DimensionTuple r : e.getOutput()) {
                int id = outputInfo.getID(r);
                outputMatrix.set(n,id,r.getValue());
                means[id].observe(r.getValue());
            }
            n++;
        }

        double[] outputMeanArr = new double[numOutputs];
        double[] outputVarArr = new double[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            outputMeanArr[i] = means[i].getMean();
            outputVarArr[i] = means[i].getVariance();
        }
        DenseVector outputMeans = DenseVector.createDenseVector(outputMeanArr);
        DenseVector outputVariances = DenseVector.createDenseVector(outputVarArr);

        // Normalize outputs
        outputMatrix.rowIntersectAndAddInPlace(outputMeans, a -> -a);
        outputMatrix.rowHadamardProductInPlace(outputVariances, a -> 1.0/a);

        Matrix featureMatrix = Matrix.aggregate(inputs, inputs.length, false);

        Kernel curKernel;
        if (optimizeKernel) {
            curKernel = null;
        } else {
            curKernel = kernel;
        }

        // Compute kernel matrix
        DenseMatrix kernelMat = curKernel.computeKernelMatrix(featureMatrix);

        // Add noise vector
        kernelMat.intersectAndAddInPlace(DenseSparseMatrix.createDiagonal(numExamples, alpha));

        // Compute cholesky
        DenseMatrix.CholeskyFactorization fact = kernelMat.choleskyFactorization().orElseThrow(() -> new IllegalStateException("Cholesky factorization failed as matrix was not positive definite, try increasing the alpha value."));
        DenseMatrix alphaMatrix = fact.solve(outputMatrix);

        String[] dimensionNames = new String[numOutputs];
        for (Regressor r : domain) {
            int id = outputInfo.getID(r);
            dimensionNames[id] = r.getNames()[0];
        }

        ModelProvenance provenance = new ModelProvenance(GaussianProcessModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        return new GaussianProcessModel("gp-regression", dimensionNames, provenance, featureIDMap, outputInfo, curKernel, featureMatrix, alphaMatrix, fact,
                outputMeans, outputVariances);
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCount;
    }

    @Override
    public void setInvocationCount(int invocationCount) {
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }

        this.trainInvocationCount = invocationCount;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
