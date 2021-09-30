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

package org.tribuo;

import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.provenance.ModelProvenance;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * A prediction model, which is used to predict outputs for unseen instances.
 * Model implementations must be serializable!
 * <p>
 * If two features map to the same id in the featureIDMap, then
 * occurrences of those features will be combined at prediction time.
 * @param <T> the type of prediction produced by the model.
 */
public abstract class Model<T extends Output<T>> implements Provenancable<ModelProvenance>, Serializable {
    private static final long serialVersionUID = 2L;

    /**
     * Used in getTopFeatures when the Model doesn't support per output feature lists.
     */
    public static final String ALL_OUTPUTS = "ALL_OUTPUTS";

    /**
     * Used to denote the bias feature in a linear model.
     */
    public static final String BIAS_FEATURE = "BIAS";

    /**
     * The model's name.
     */
    protected String name;

    /**
     * The model provenance.
     */
    protected final ModelProvenance provenance;

    /**
     * The cached toString of the model provenance.
     * <p>
     * Mostly cached so it appears in the serialized output and can be read by grepping the binary.
     */
    protected final String provenanceOutput;

    /**
     * The features this model knows about.
     */
    protected final ImmutableFeatureMap featureIDMap;

    /**
     * The outputs this model predicts.
     */
    protected final ImmutableOutputInfo<T> outputIDInfo;

    /**
     * Does this model generate probability distributions in the output.
     */
    protected final boolean generatesProbabilities;

    /**
     * Constructs a new model, storing the supplied fields.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The features.
     * @param outputIDInfo The possible outputs.
     * @param generatesProbabilities Does this model emit probabilistic outputs.
     */
    protected Model(String name, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, boolean generatesProbabilities) {
        this.name = name;
        this.provenance = provenance;
        this.provenanceOutput = provenance.toString();
        this.featureIDMap = featureIDMap;
        this.outputIDInfo = outputIDInfo;
        this.generatesProbabilities = generatesProbabilities;
    }

    /**
     * Returns the model name.
     * @return The model name.
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the model name.
     * @param name The new model name.
     */
    public void setName(String name) {
        this.name = name;
    }
 
    @Override
    public ModelProvenance getProvenance() {
        return provenance;
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
        return outputIDInfo;
    }

    /**
     * Does this model generate probabilistic predictions.
     * @return True if the model generates probabilistic predictions.
     */
    public boolean generatesProbabilities() {
        return generatesProbabilities;
    }

    /**
     * Validates that this Model does in fact support the supplied output type.
     * <p>
     * As the output type is erased at runtime, deserialising a Model is an unchecked
     * operation. This method allows the user to check that the deserialised model is
     * of the appropriate type, rather than seeing if {@link Model#predict} throws a {@link ClassCastException}
     * when called.
     * </p>
     * @param clazz The class object to verify the output type against.
     * @return True if the output type is assignable to the class object type, false otherwise.
     */
    public boolean validate(Class<? extends Output<?>> clazz) {
        Set<T> domain = outputIDInfo.getDomain();
        boolean output = true;
        for (T type : domain) {
            output &= clazz.isInstance(type);
        }
        return output;
    }

    /**
     * Uses the model to predict the output for a single example.
     * <p>
     * predict does not mutate the example.
     * <p>
     * Throws {@link IllegalArgumentException} if the example has no features
     * or no feature overlap with the model.
     * @param example the example to predict.
     * @return the result of the prediction.
     */
    public abstract Prediction<T> predict(Example<T> example);
    
    /**
     * Uses the model to predict the output for multiple examples.
     * <p>
     * Throws {@link IllegalArgumentException} if the examples have no features
     * or no feature overlap with the model.
     * @param examples the examples to predict.
     * @return the results of the prediction, in the same order as the 
     * examples.
     */
    public List<Prediction<T>> predict(Iterable<Example<T>> examples) {
        return innerPredict(examples);
    }

    /**
     * Uses the model to predict the outputs for multiple examples contained in
     * a data set.
     * <p>
     * Throws {@link IllegalArgumentException} if the examples have no features
     * or no feature overlap with the model.
     * @param examples the data set containing the examples to predict.
     * @return the results of the predictions, in the same order as the 
     * Dataset provides the examples.
     */
    public List<Prediction<T>> predict(Dataset<T> examples) {
        return innerPredict(examples);
    }

    /**
     * Called by the base implementations of {@link Model#predict(Iterable)} and {@link Model#predict(Dataset)}.
     * @param examples The examples to predict.
     * @return The results of the predictions, in the same order as the examples.
     */
    protected List<Prediction<T>> innerPredict(Iterable<Example<T>> examples) {
        List<Prediction<T>> predictions = new ArrayList<>();
        for (Example<T> example : examples) {
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
    public abstract Map<String,List<Pair<String,Double>>> getTopFeatures(int n);

    /**
     * Generates an excuse for an example.
     * <p>
     * This attempts to explain a classification result. Generating an excuse may be quite an expensive operation.
     * <p>
     * This excuse either contains per class information or an entry with key Model.ALL_OUTPUTS.
     * <p>
     * The optional is empty if the model does not provide excuses.
     * @param example The input example.
     * @return An optional excuse object. The optional is empty if this model does not provide excuses.
     */
    public abstract Optional<Excuse<T>> getExcuse(Example<T> example);

    /**
     * Generates an excuse for each example.
     * <p>
     * This may be an expensive operation, and probably should be overridden in subclasses for performance reasons.
     * <p>
     * These excuses either contain per class information or an entry with key Model.ALL_OUTPUTS.
     * <p>
     * The optional is empty if the model does not provide excuses.
     * @param examples An iterable of examples
     * @return A optional list of excuses. The Optional is empty if this model does not provide excuses.
     */
    public Optional<List<Excuse<T>>> getExcuses(Iterable<Example<T>> examples) {
        List<Excuse<T>> excuses = new ArrayList<>();
        for (Example<T> e : examples) {
            Optional<Excuse<T>> excuse = getExcuse(e);
            if (excuse.isPresent()) {
                excuses.add(excuse.get());
            } else {
                return Optional.empty();
            }
        }
        return Optional.of(excuses);
    }

    /**
     * Copies a model, returning a deep copy of any mutable state, and a shallow copy otherwise.
     * @return A copy of the model.
     */
    public Model<T> copy() {
        List<ObjectMarshalledProvenance> omp = ProvenanceUtil.marshalProvenance(provenance);
        ModelProvenance provenanceCopy = (ModelProvenance) ProvenanceUtil.unmarshalProvenance(omp);
        return copy(name,provenanceCopy);
    }

    /**
     * Copies a model, replacing its provenance and name with the supplied values.
     * <p>
     * Used to provide the provenance removal functionality.
     * @param newName The new name.
     * @param newProvenance The new provenance.
     * @return A copy of the model.
     */
    protected abstract Model<T> copy(String newName, ModelProvenance newProvenance);

    @Override
    public String toString() {
        if (name != null && !name.isEmpty()) {
            return name + " - " + provenanceOutput;
        } else {
            return provenanceOutput;
        }
    }

    /**
     * Casts the model to the specified output type, assuming it is valid.
     * <p>
     * If it's not valid, throws {@link ClassCastException}.
     * @param inputModel The model to cast.
     * @param outputType The output type to cast to.
     * @param <T> The output type.
     * @return The model cast to the correct value.
     */
    public static <T extends Output<T>> Model<T> castModel(Model<?> inputModel, Class<T> outputType) {
        if (inputModel.validate(outputType)) {
            @SuppressWarnings("unchecked") // guarded by validate
            Model<T> castedModel = (Model<T>) inputModel;
            return castedModel;
        } else {
            throw new ClassCastException("Attempted to cast model to " + outputType.getName() + " which is not valid for model " + inputModel.toString());
        }
    }
    
}
