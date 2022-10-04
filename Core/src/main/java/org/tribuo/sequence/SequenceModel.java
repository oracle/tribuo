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

package org.tribuo.sequence;

import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.SequenceModelProto;
import org.tribuo.provenance.ModelProvenance;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * A prediction model, which is used to predict outputs for unseen instances.
 * @param <T> the type of the outputs used to train the model.
 */
public abstract class SequenceModel<T extends Output<T>> implements ProtoSerializable<SequenceModelProto>, Provenancable<ModelProvenance>, Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * The model name.
     */
    protected String name;

    private final ModelProvenance provenance;

    /**
     * The toString of the model provenance.
     */
    protected final String provenanceOutput;

    /**
     * The feature domain.
     */
    protected final ImmutableFeatureMap featureIDMap;

    /**
     * The output domain.
     */
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

    @Override
    public SequenceModelProto serialize() {
        throw new UnsupportedOperationException("The default implementation of SequenceModel.serialize() must be overridden to support protobuf serialization.");
    }

    /**
     * Serializes this model to a {@link SequenceModelProto} and writes it to the supplied path.
     * @param path The path to write to.
     * @throws IOException If the path could not be written to.
     */
    public void serializeToFile(Path path) throws IOException {
        try (OutputStream os = new BufferedOutputStream(Files.newOutputStream(path))) {
            serializeToStream(os);
        }
    }

    /**
     * Serializes this model to a {@link SequenceModelProto} and writes it to the supplied output stream.
     * <p>
     * Does not close the stream.
     * @param stream The output stream to write to.
     * @throws IOException If the stream could not be written to.
     */
    public void serializeToStream(OutputStream stream) throws IOException {
        SequenceModelProto proto = serialize();
        proto.writeTo(stream);
    }

    /**
     * Deserializes the model from the supplied protobuf.
     * @param proto The protobuf to deserialize.
     * @return The model.
     */
    public static SequenceModel<?> deserialize(SequenceModelProto proto) {
        return ProtoUtil.deserialize(proto);
    }

    /**
     * Reads an instance of {@link SequenceModelProto} from the supplied path and deserializes it.
     * @param path The path to read.
     * @return The deserialized model.
     * @throws IOException If the path could not be read from, or the parsing failed.
     */
    public static SequenceModel<?> deserializeFromFile(Path path) throws IOException {
        try (InputStream is = new BufferedInputStream(Files.newInputStream(path))) {
            return deserializeFromStream(is);
        }
    }

    /**
     * Reads an instance of {@link SequenceModelProto} from the supplied input stream and deserializes it.
     * <p>
     * Does not close the stream.
     * @param is The input stream to read.
     * @return The deserialized model.
     * @throws IOException If the stream could not be read from, or the parsing failed.
     */
    public static SequenceModel<?> deserializeFromStream(InputStream is) throws IOException {
        SequenceModelProto proto = SequenceModelProto.parseFrom(is);
        return deserialize(proto);
    }

    /**
     * Constructs the data carrier for serialization.
     * @return The serialization data carrier.
     */
    protected ModelDataCarrier<T> createDataCarrier() {
        return new ModelDataCarrier<>(name,provenance,featureIDMap,outputIDMap,false,provenance.getTribuoVersion());
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
     * all features should be returned for each class, unless the model cannot score its features.
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

    /**
     * Casts the model to the specified output type, assuming it is valid.
     * If it's not valid, throws {@link ClassCastException}.
     * <p>
     * This method is intended for use on a deserialized model to restore its
     * generic type in a safe way.
     * @param outputType The output type to cast to.
     * @param <U> The output type.
     * @return The model cast to the correct value.
     */
    public <U extends Output<U>> SequenceModel<U> castModel(Class<U> outputType) {
        if (validate(outputType)) {
            @SuppressWarnings("unchecked") // guarded by validate
            SequenceModel<U> castedModel = (SequenceModel<U>) this;
            return castedModel;
        } else {
            throw new ClassCastException("Attempted to cast sequence model to " + outputType.getName() + " which is not valid for model " + this.toString());
        }
    }

}
