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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
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

/**
 * Image transformer. Assumes the feature id numbers are linearised ids of the form
 * [0,0,0] = 0, [1,0,0] = 1, ..., [i,0,0] = i, [0,1,0] = i+1, ..., [i,j,0] = i*j, ...
 * [0,0,1] = (i*j)+1, ..., [i,j,k] = i*j*k.
 */
public class ImageTransformer<T extends Output<T>> implements ExampleTransformer<T> {
    private static final long serialVersionUID = 1L;

    @Config(mandatory=true,description="Image width.")
    private int width;

    @Config(mandatory=true,description="Image height.")
    private int height;

    @Config(mandatory=true,description="Number of channels.")
    private int channels;

    /**
     * For olcut.
     */
    private ImageTransformer() {}

    public ImageTransformer(int width, int height, int channels) {
        if (width < 1 || height < 1 || channels < 1) {
            throw new IllegalArgumentException("Inputs must be positive integers, found ["+width+","+height+","+channels+"]");
        }
        this.width = width;
        this.height = height;
        this.channels = channels;
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (width < 1 || height < 1 || channels < 1) {
            throw new PropertyException("","Inputs must be positive integers, found ["+width+","+height+","+channels+"]");
        }
    }

    /**
     * Transform implicitly pads unseen values with zero.
     * @param example The example to transform.
     * @param featureIDMap The feature id mapping to use.
     * @return A 3d tensor, (width, height, channels) for this example.
     */
    @Override
    public Tensor<?> transform(Example<T> example, ImmutableFeatureMap featureIDMap) {
        float[][][][] image = new float[1][][][];
        image[0] = innerTransform(example,featureIDMap);
        return Tensor.create(image);
    }

    /**
     * Actually performs the transformation. Implicitly pads unseen values
     * with zero.
     * @param example The example to transform.
     * @param featureIDMap The feature id mapping to use.
     * @return A 3d array, (width,height,channels) representing the example.
     */
    float[][][] innerTransform(Example<T> example, ImmutableFeatureMap featureIDMap) {
        float[][][] image = new float[width][height][channels];

        for (Feature f : example) {
            int id = featureIDMap.getID(f.getName());
            int curWidth = id % width;
            int curHeight = (id / width) % height;
            int curChannel = id / (width * height);
            image[curWidth][curHeight][curChannel] = (float) f.getValue();
        }

        return image;
    }

    /**
     * Actually performs the transformation. Implicitly pads unseen values
     * with zero.
     * @param vector The vector to transform.
     * @return A 3d array, (width,height,channels) representing the vector.
     */
    float[][][] innerTransform(SparseVector vector) {
        float[][][] image = new float[width][height][channels];

        for (VectorTuple f : vector) {
            int id = f.index;
            int curWidth = id % width;
            int curHeight = (id / width) % height;
            int curChannel = id / (width * height);
            image[curWidth][curHeight][curChannel] = (float) f.value;
        }

        return image;
    }

    /**
     * Transform implicitly pads unseen values with zero.
     * <p>
     * Converts a batch of examples into a Tensor.
     * @param examples The examples to transform.
     * @param featureIDMap The feature id mapping to use.
     * @return A 4d tensor, (batch-id, width, height, channels) for this example.
     */
    @Override
    public Tensor<?> transform(List<Example<T>> examples, ImmutableFeatureMap featureIDMap) {
        float[][][][] image = new float[examples.size()][][][];

        int i = 0;
        for (Example<T> example : examples) {
            image[i] = innerTransform(example,featureIDMap);
            i++;
        }

        return Tensor.create(image);
    }

    @Override
    public Tensor<?> transform(SparseVector vector) {
        float[][][][] image = new float[1][][][];
        image[0] = innerTransform(vector);
        return Tensor.create(image);
    }

    @Override
    public Tensor<?> transform(List<SparseVector> vectors) {
        float[][][][] image = new float[vectors.size()][][][];

        int i = 0;
        for (SparseVector vector : vectors) {
            image[i] = innerTransform(vector);
            i++;
        }

        return Tensor.create(image);
    }

    @Override
    public String toString() {
        return "ImageTransformer(width="+width+",height="+height+",channels="+channels+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"ExampleTransformer");
    }
}
