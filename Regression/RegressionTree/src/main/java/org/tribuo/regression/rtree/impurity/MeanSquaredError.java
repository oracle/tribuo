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

package org.tribuo.regression.rtree.impurity;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

import java.util.List;

/**
 * Measures the mean squared error over a set of inputs.
 * <p>
 * Used to calculate the impurity of a regression node.
 */
public class MeanSquaredError implements RegressorImpurity {

    @Override
    public double impurity(float[] targets, float[] weights) {
        float weightedSum = 0.0f;
        float weightSum = 0.0f;
        for (int i = 0; i < targets.length; i++) {
            weightedSum += targets[i]*weights[i];
            weightSum += weights[i];
        }
        float mean = weightedSum / weightSum;

        float squaredError = 0.0f;

        for (int i = 0; i < targets.length; i++) {
            float error = mean - targets[i];
            squaredError += error*error*weights[i];
        }
        return squaredError / weightSum;
    }

    @Override
    public ImpurityTuple impurityTuple(int[] indices, int indicesLength, float[] targets, float[] weights) {
        if (indicesLength == 1) {
            return new ImpurityTuple(0.0f,weights[indices[0]]);
        } else {
            float weightedSquaredSum = 0.0f;
            float weightedSum = 0.0f;
            float weightSum = 0.0f;
            for (int i = 0; i < indicesLength; i++) {
                int idx = indices[i];
                float weight = weights[idx];
                float target = targets[idx];
                float curVal = target * weight;
                weightedSum += curVal;
                weightedSquaredSum += curVal * target;
                weightSum += weight;
            }
            float mean = weightedSum / weightSum;
            return new ImpurityTuple((weightedSquaredSum / weightSum) - (mean*mean),weightSum);
        }
    }

    @Override
    public ImpurityTuple impurityTuple(List<int[]> indices, float[] targets, float[] weights) {
        float weightedSquaredSum = 0.0f;
        float weightedSum = 0.0f;
        float weightSum = 0.0f;
        for (int[] curIndices : indices) {
            for (int i = 0; i < curIndices.length; i++) {
                int idx = curIndices[i];
                float weight = weights[idx];
                float target = targets[idx];
                float curVal = target * weight;
                weightedSum += curVal;
                weightedSquaredSum += curVal * target;
                weightSum += weight;
            }
        }

        float mean = weightedSum / weightSum;
        return new ImpurityTuple((weightedSquaredSum / weightSum) - (mean*mean),weightSum);
    }

    @Override
    public String toString() {
        return "MeanSquaredError";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"RegressorImpurity");
    }
}
