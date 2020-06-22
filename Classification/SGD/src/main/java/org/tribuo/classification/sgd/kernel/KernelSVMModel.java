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

package org.tribuo.classification.sgd.kernel;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.math.kernel.Kernel;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The inference time version of a kernel model trained using Pegasos.
 * <p>
 * See:
 * <pre>
 * Shalev-Shwartz S, Singer Y, Srebro N, Cotter A
 * "Pegasos: Primal Estimated Sub-Gradient Solver for SVM"
 * Mathematical Programming, 2011.
 * </pre>
 */
public class KernelSVMModel extends Model<Label> {
    private static final long serialVersionUID = 2L;

    private final Kernel kernel;
    private final SparseVector[] supportVectors;
    private final DenseMatrix weights;

    KernelSVMModel(String name, ModelProvenance description,
                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> labelIDMap,
                          Kernel kernel, SparseVector[] supportVectors, DenseMatrix weights) {
        super(name, description, featureIDMap, labelIDMap, false);
        this.kernel = kernel;
        this.supportVectors = supportVectors;
        this.weights = weights;
    }

    /**
     * Returns the number of support vectors used.
     * @return The number of support vectors.
     */
    public int getNumberOfSupportVectors() {
        return supportVectors.length;
    }

    @Override
    public Prediction<Label> predict(Example<Label> example) {
        SparseVector features = SparseVector.createSparseVector(example,featureIDMap,true);
        // Due to bias feature
        if (features.numActiveElements() == 1) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }
        double[] scores = new double[supportVectors.length];
        for (int i = 0; i < scores.length; i++) {
            scores[i] = kernel.similarity(features,supportVectors[i]);
        }
        DenseVector scoreVector = DenseVector.createDenseVector(scores);
        DenseVector prediction = weights.leftMultiply(scoreVector);

        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        Map<String,Label> predMap = new LinkedHashMap<>();
        for (int i = 0; i < prediction.size(); i++) {
            String labelName = outputIDInfo.getOutput(i).getLabel();
            Label label = new Label(labelName, prediction.get(i));
            predMap.put(labelName, label);
            if (label.getScore() > maxScore) {
                maxScore = label.getScore();
                maxLabel = label;
            }
        }
        return new Prediction<>(maxLabel, predMap, features.numActiveElements(), example, generatesProbabilities);
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    public Optional<Excuse<Label>> getExcuse(Example<Label> example) {
        return Optional.empty();
    }

    @Override
    protected KernelSVMModel copy(String newName, ModelProvenance newProvenance) {
        SparseVector[] vectorCopies = new SparseVector[supportVectors.length];
        for (int i = 0; i < vectorCopies.length; i++) {
            vectorCopies[i] = supportVectors[i].copy();
        }
        return new KernelSVMModel(newName,newProvenance,featureIDMap,outputIDInfo,kernel,vectorCopies,new DenseMatrix(weights));
    }
}
