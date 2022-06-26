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

package org.tribuo.transform;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Wraps a {@link Model} with it's {@link TransformerMap} so all {@link Example}s are transformed
 * appropriately before the model makes predictions.
 * <p>
 * If the densify flag is set, densifies all incoming data before transforming it.
 * <p>
 * Transformations only operate on observed values. To operate on implicit zeros then
 * first call {@link MutableDataset#densify} on the datasets.
 */
public class TransformedModel<T extends Output<T>> extends Model<T> {
    private static final long serialVersionUID = 1L;

    private final Model<T> innerModel;

    private final TransformerMap transformerMap;

    private final boolean densify;

    private ArrayList<String> featureNames;

    TransformedModel(ModelProvenance modelProvenance, Model<T> innerModel, TransformerMap transformerMap, boolean densify) {
        super(innerModel.getName(),
              modelProvenance,
              innerModel.getFeatureIDMap(),
              innerModel.getOutputIDInfo(),
              innerModel.generatesProbabilities());
        this.innerModel = innerModel;
        this.transformerMap = transformerMap;
        this.densify = densify;
        this.featureNames = new ArrayList<>(featureIDMap.keySet());
        Collections.sort(featureNames);
    }

    /**
     * Gets the transformers that this model applies to each example.
     * <p>
     * Note if you use these transformers to modify some data, and then
     * feed that data to this model, the data will be transformed twice
     * and this is not what you want.
     * @return The transformers.
     */
    public TransformerMap getTransformerMap() {
        return transformerMap;
    }

    /**
     * Gets the inner model to allow access to any class specific methods
     * that model contains (e.g., to examine cluster centroids).
     * <p>
     * Note that this model expects all examples to have been transformed using
     * the transformer map, which can be extracted with {@link #getTransformerMap}.
     * @return The inner model.
     */
    public Model<T> getInnerModel() {
        return innerModel;
    }

    /**
     * Returns true if the model densifies the feature space before applying the transformations.
     * @return True if the transforms operate on the dense feature space.
     */
    public boolean getDensify() {
        return densify;
    }

    @Override
    public Prediction<T> predict(Example<T> example) {
        Example<T> transformedExample;
        if (densify) {
            transformedExample = transformerMap.transformExample(example,featureNames);
        } else {
            transformedExample = transformerMap.transformExample(example);
        }
        return innerModel.predict(transformedExample);
    }

    @Override
    public List<Prediction<T>> predict(Dataset<T> examples) {
        Dataset<T> transformedDataset = transformerMap.transformDataset(examples,densify);

        List<Prediction<T>> predictions = new ArrayList<>();
        for (Example<T> example : transformedDataset) {
            predictions.add(innerModel.predict(example));
        }

        return predictions;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return innerModel.getTopFeatures(n);
    }

    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        Example<T> transformedExample = transformerMap.transformExample(example);
        return innerModel.getExcuse(transformedExample);
    }

    @Override
    protected TransformedModel<T> copy(String name, ModelProvenance newProvenance) {
        return new TransformedModel<>(newProvenance,innerModel,transformerMap,densify);
    }
}
