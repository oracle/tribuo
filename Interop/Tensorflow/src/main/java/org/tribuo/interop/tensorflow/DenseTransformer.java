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

package org.tribuo.interop.tensorflow;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Output;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorTuple;
import org.tensorflow.Tensor;

import java.util.List;
import java.util.logging.Logger;

/**
 * Converts a sparse example into a dense float vector, then wraps it in a {@link Tensor}.
 */
public class DenseTransformer<T extends Output<T>> implements ExampleTransformer<T> {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(DenseTransformer.class.getName());

    /**
     * Feature size beyond which a warning is generated (as ONNX requires dense features and large feature spaces are memory hungry).
     */
    public static final int THRESHOLD = 1000000;

    /**
     * Number of times the feature size warning should be printed.
     */
    public static final int WARNING_THRESHOLD = 10;

    private int warningCount = 0;

    public DenseTransformer() { }

    float[] innerTransform(Example<T> example, ImmutableFeatureMap featureIDMap) {
        if ((warningCount < WARNING_THRESHOLD) && (featureIDMap.size() > THRESHOLD)) {
            logger.warning("Large dense example requested, featureIDMap.size() = " + featureIDMap.size() + ", example.size() = " + example.size());
            warningCount++;
        }
        float[] output = new float[featureIDMap.size()];

        for (Feature f : example) {
            int id = featureIDMap.getID(f.getName());
            if (id > -1) {
                output[id] = (float) f.getValue();
            }
        }

        return output;
    }

    float[] innerTransform(SparseVector vector) {
        if ((warningCount < WARNING_THRESHOLD) && (vector.size() > THRESHOLD)) {
            logger.warning("Large dense example requested, dimension = " + vector.size() + ", numActiveElements = " + vector.numActiveElements());
            warningCount++;
        }
        float[] output = new float[vector.size()];

        for (VectorTuple f : vector) {
            output[f.index] = (float) f.value;
        }

        return output;
    }

    @Override
    public Tensor<?> transform(Example<T> example, ImmutableFeatureMap featureIDMap) {
        float[][] output = new float[1][];
        output[0] = innerTransform(example,featureIDMap);
        return Tensor.create(output);
    }

    @Override
    public Tensor<?> transform(List<Example<T>> examples, ImmutableFeatureMap featureIDMap) {
        float[][] output = new float[examples.size()][];

        int i = 0;
        for (Example<T> example : examples) {
            output[i] = innerTransform(example,featureIDMap);
            i++;
        }

        return Tensor.create(output);
    }

    @Override
    public Tensor<?> transform(SparseVector vector) {
        float[][] output = new float[1][];
        output[0] = innerTransform(vector);
        return Tensor.create(output);
    }

    @Override
    public Tensor<?> transform(List<SparseVector> vectors) {
        float[][] output = new float[vectors.size()][];

        int i = 0;
        for (SparseVector vector : vectors) {
            output[i] = innerTransform(vector);
            i++;
        }

        return Tensor.create(output);
    }

    @Override
    public String toString() {
        return "DenseTransformer()";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"ExampleTransformer");
    }
}
