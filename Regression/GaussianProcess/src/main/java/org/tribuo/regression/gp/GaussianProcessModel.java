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

package org.tribuo.regression.gp;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.math.kernel.Kernel;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

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
public final class GaussianProcessModel extends Model<Regressor> {

    private final String[] dimensionNames;
    private final Kernel kernel;
    private final Matrix features;
    private final DenseMatrix alpha;
    private final DenseMatrix.CholeskyFactorization fact;
    private final DenseVector outputMeans;
    private final DenseVector outputVariances;

    GaussianProcessModel(String name, String[] dimensionNames, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputInfo, Kernel kernel, Matrix featureMatrix, DenseMatrix alphaMatrix, DenseMatrix.CholeskyFactorization fact, DenseVector outputMeans, DenseVector outputVariances) {
        super(name, provenance, featureIDMap, outputInfo, false);
        this.dimensionNames = dimensionNames;
        this.kernel = kernel;
        this.features = featureMatrix;
        this.alpha = alphaMatrix;
        this.fact = fact;
        this.outputMeans = outputMeans;
        this.outputVariances = outputVariances;
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        SGDVector vec;
        if (example.size() == featureIDMap.size()) {
            vec =  DenseVector.createDenseVector(example, featureIDMap, false);
        } else {
            vec = SparseVector.createSparseVector(example, featureIDMap, false);
        }
        DenseVector sim = kernel.computeSimilarityVector(vec, features);
        DenseVector meanPred = alpha.rightMultiply(sim);
        meanPred.hadamardProductInPlace(outputVariances);
        meanPred.intersectAndAddInPlace(outputMeans);
        return new Prediction<>(new Regressor(dimensionNames,meanPred.toArray()), vec.numActiveElements(), example);
    }

    @Override
    protected List<Prediction<Regressor>> innerPredict(Iterable<Example<Regressor>> examples) {
        List<Prediction<Regressor>> predictions = new ArrayList<>();
        for (Example<Regressor> example : examples) {
            predictions.add(predict(example));
        }
        return predictions;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    public Optional<Excuse<Regressor>> getExcuse(Example<Regressor> example) {
        return Optional.empty();
    }

    @Override
    protected Model<Regressor> copy(String newName, ModelProvenance newProvenance) {
        return new GaussianProcessModel(newName, Arrays.copyOf(dimensionNames, dimensionNames.length), newProvenance,
                featureIDMap, outputIDInfo, kernel, features.copy(), alpha.copy(), fact.copy(), outputMeans.copy(),
                outputVariances.copy());
    }
}
