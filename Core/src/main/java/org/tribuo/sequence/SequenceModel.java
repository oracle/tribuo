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

package org.tribuo.sequence;

import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.provenance.ModelProvenance;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * A prediction model, which is used to predict outputs for unseen instances.
 * @param <T> the type of the outputs used to train the model.
 */
public abstract class SequenceModel<T extends Output<T>> implements Provenancable<ModelProvenance>, Serializable {
    private static final long serialVersionUID = 1L;

    protected String name;

    private final ModelProvenance provenance;

    protected final String provenanceOutput;

    protected final ImmutableFeatureMap featureIDMap;

    protected final ImmutableOutputInfo<T> outputIDMap;

    /**
     * Builds a SequenceModel.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDMap The output domain.
     */
    public SequenceModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDMap) {
        this.name = name;
        this.provenance = provenance;
        this.provenanceOutput = provenance.toString();
        this.featureIDMap = featureIDMap;
        this.outputIDMap = outputIDMap;
    }

    /**
     * Validates that this Model does in fact support the supplied output type.
     * <p>
     * As the output type is erased at runtime, deserialising a Model is an unchecked
     * operation. This method allows the user to check that the deserialised model is
     * of the appropriate type, rather than seeing if {@link SequenceModel#predict} throws a {@link ClassCastException}
     * when called.
     * </p>
     * @param clazz The class object to verify the output type against.
     * @return True if the output type is assignable to the class object type, false otherwise.
     */
    public boolean validate(Class<? extends Output<?>> clazz) {
        Set<T> domain = outputIDMap.getDomain();
        boolean output = true;
        for (T type : domain) {
            output &= clazz.isInstance(type);
        }
        return output;
    }

    /**
     * Gets the model name.
     * @return The model name.
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the model name.
     * @param name The model name.
     */
    public void setName(String name) {
        this.name = name;
    }

    @Override
    public ModelProvenance getProvenance() {
        return provenance;
    }

    @Override
    public String toString() {
        if (name != null && !name.isEmpty()) {
            return name + " - " + provenanceOutput;
        } else {
            return provenanceOutput;
        }
    }

    /**
     * Gets the feature domain.
     * @return The feature domain.
     */
    public ImmutableFeatureMap getFeatureIDMap() {
        return featureIDMap;
    }

    /**
     * Gets the output domain.
     * @return The output domain.
     */
    public ImmutableOutputInfo<T> getOutputIDInfo() {
        return outputIDMap;
    }

    /**
     * Uses the model to predict the output for a single example.
     * @param example the example to predict.
     * @return the result of the prediction.
     */
    public abstract List<Prediction<T>> predict(SequenceExample<T> example);
    
    /**
     * Uses the model to predict the output for multiple examples.
     * @param examples the examples to predict.
     * @return the results of the prediction, in the same order as the 
     * examples.
     */
    public List<List<Prediction<T>>> predict(Iterable<SequenceExample<T>> examples) {
        List<List<Prediction<T>>> predictions = new ArrayList<>();
        for(SequenceExample<T> example : examples) {
            predictions.add(predict(example));
        }
        return predictions;
    }

    /**
     * Uses the model to predict the labels for multiple examples contained in
     * a data set.
     * @param examples the data set containing the examples to predict.
     * @return the results of the predictions, in the same order as the 
     * data set generates the example.
     */
    public List<List<Prediction<T>>> predict(SequenceDataset<T> examples) {
        List<List<Prediction<T>>> predictions = new ArrayList<>();
        for (SequenceExample<T> example : examples) {
            predictions.add(predict(example));
        }
        return predictions;
    }
    
    /**
     * Gets the top {@code n} features associated with this model.
     * <p>
     * If the model does not produce per output feature lists, it returns
     * a map with a single element with key Model.ALL_OUTPUTS.
     * </p>
     * <p>
     * If the model cannot describe it's top features then it returns {@link java.util.Collections#emptyMap}.
     * </p>
     * @param n the number of features to return. If this value is less than 0,
     * all features should be returned for each class, unless the model cannot score it's features.
     * @return a map from string outputs to an ordered list of pairs of
     * feature names and weights associated with that feature in the model
     */
    public abstract Map<String, List<Pair<String, Double>>> getTopFeatures(int n);

    /**
     * Extracts a list of the predicted outputs from the list of prediction objects.
     * @param predictions The predictions.
     * @param <T> The prediction type.
     * @return A list of predicted outputs.
     */
    public static <T extends Output<T>> List<T> toMaxLabels(List<Prediction<T>> predictions) {
        return predictions.stream().map(Prediction::getOutput).collect(Collectors.toList());
    }
}
