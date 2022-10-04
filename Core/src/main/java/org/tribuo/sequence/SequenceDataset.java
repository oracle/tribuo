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
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.FeatureMap;
import org.tribuo.ImmutableDataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.OutputInfo;
import org.tribuo.Tribuo;
import org.tribuo.impl.DatasetDataCarrier;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.SequenceDatasetProto;
import org.tribuo.protos.core.SequenceExampleProto;
import org.tribuo.provenance.DataProvenance;
import org.tribuo.provenance.DatasetProvenance;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A class for sets of data, which are used to train and evaluate classifiers.
 * <p>
 * Subclass either {@link MutableSequenceDataset} or {@link ImmutableSequenceDataset} rather than this class.
 *
 * @param <T> the type of the outputs in the data set.
 */
public abstract class SequenceDataset<T extends Output<T>> implements Iterable<SequenceExample<T>>, ProtoSerializable<SequenceDatasetProto>, Provenancable<DatasetProvenance>, Serializable {
    private static final Logger logger = Logger.getLogger(SequenceDataset.class.getName());
    private static final long serialVersionUID = 2L;

    /**
     * A factory for making {@link OutputInfo} and {@link Output} of the appropriate type.
     */
    protected final OutputFactory<T> outputFactory;

    /**
     * The data in this data set.
     */
    protected final List<SequenceExample<T>> data = new ArrayList<>();

    /**
     * The version of Tribuo which created this dataset.
     */
    protected final String tribuoVersion;

    /**
     * The provenance of the data source, extracted on construction.
     */
    protected final DataProvenance sourceProvenance;

    /**
     * Constructs a sequence dataset using the current Tribuo version.
     * @param sourceProvenance The provenance.
     * @param outputFactory The output factory.
     */
    protected SequenceDataset(DataProvenance sourceProvenance, OutputFactory<T> outputFactory) {
        this(sourceProvenance, outputFactory, Tribuo.VERSION);
    }

    /**
     * Constructs a sequence dataset.
     * @param sourceProvenance The provenance.
     * @param outputFactory The output factory.
     * @param tribuoVersion The Tribuo version string.
     */
    protected SequenceDataset(DataProvenance sourceProvenance, OutputFactory<T> outputFactory, String tribuoVersion) {
        this.sourceProvenance = sourceProvenance;
        this.outputFactory = outputFactory;
        this.tribuoVersion = tribuoVersion;
    }

    /**
     * Returns the description of the source provenance.
     * @return The source provenance in text form.
     */
    public String getSourceDescription() {
        return "SequenceDataset(source=" + sourceProvenance.toString() + ")";
    }

    /**
     * Returns an unmodifiable view on the data.
     * @return The data.
     */
    public List<SequenceExample<T>> getData() {
        return Collections.unmodifiableList(data);
    }

    /**
     * Returns the source provenance.
     * @return The source provenance.
     */
    public DataProvenance getSourceProvenance() {
        return sourceProvenance;
    }

    /**
     * Gets the set of labels that occur in the examples in this dataset.
     *
     * @return the set of labels that occur in the examples in this dataset.
     */
    public abstract Set<T> getOutputs();

    /**
     * Gets the example at the specified index, or throws IllegalArgumentException if
     * the index is out of bounds.
     * @param index The index.
     * @return The example at that index.
     */
    public SequenceExample<T> getExample(int index) {
        if ((index < 0) || (index >= size())) {
            throw new IllegalArgumentException("Example index " + index + " is out of bounds.");
        }
        return data.get(index);
    }

    /**
     * Returns a view on this SequenceDataset which aggregates all
     * the examples and ignores the sequence structure.
     *
     * @return A flattened view on this dataset.
     */
    public Dataset<T> getFlatDataset() {
        return new FlatDataset<>(this);
    }

    /**
     * Gets the size of the data set.
     *
     * @return the size of the data set.
     */
    public int size() {
        return data.size();
    }

    /**
     * An immutable view on the output info in this dataset.
     * @return The output info.
     */
    public abstract ImmutableOutputInfo<T> getOutputIDInfo();

    /**
     * The output info in this dataset.
     * @return The output info.
     */
    public abstract OutputInfo<T> getOutputInfo();

    /**
     * An immutable view on the feature map.
     * @return The feature map.
     */
    public abstract ImmutableFeatureMap getFeatureIDMap();

    /**
     * The feature map.
     * @return The feature map.
     */
    public abstract FeatureMap getFeatureMap();

    /**
     * Gets the output factory.
     * @return The output factory.
     */
    public OutputFactory<T> getOutputFactory() {
        return outputFactory;
    }

    @Override
    public Iterator<SequenceExample<T>> iterator() {
        return data.iterator();
    }

    @Override
    public String toString() {
        return "SequenceDataset(source=" + sourceProvenance.toString() + ")";
    }

    /**
     * Validates that this SequenceDataset does in fact contain the supplied output type.
     * <p>
     * As the output type is erased at runtime, deserialising a SequenceDataset is an unchecked
     * operation. This method allows the user to check that the deserialised dataset is
     * of the appropriate type, rather than seeing if the Dataset throws a {@link ClassCastException}
     * when used.
     * @param clazz The class object to verify the output type against.
     * @return True if the output type is assignable to the class object type, false otherwise.
     */
    public boolean validate(Class<? extends Output<?>> clazz) {
        Set<T> domain = getOutputInfo().getDomain();
        boolean output = true;
        for (T type : domain) {
            output &= clazz.isInstance(type);
        }
        return output;
    }

    /**
     * Casts the dataset to the specified output type, assuming it is valid.
     * <p>
     * If it's not valid, throws {@link ClassCastException}.
     * @param inputDataset The model to cast.
     * @param outputType The output type to cast to.
     * @param <T> The output type.
     * @return The model cast to the correct value.
     */
    public static <T extends Output<T>> SequenceDataset<T> castDataset(SequenceDataset<?> inputDataset, Class<T> outputType) {
        if (inputDataset.validate(outputType)) {
            @SuppressWarnings("unchecked") // guarded by validate
            SequenceDataset<T> castedModel = (SequenceDataset<T>) inputDataset;
            return castedModel;
        } else {
            throw new ClassCastException("Attempted to cast dataset to " + outputType.getName() + " which is not valid for dataset " + inputDataset.toString());
        }
    }

    /**
     * Deserializes a sequence dataset proto into a sequence dataset.
     * @param sequenceProto The proto to deserialize.
     * @return The sequence dataset.
     */
    public static SequenceDataset<?> deserialize(SequenceDatasetProto sequenceProto) {
        return ProtoUtil.deserialize(sequenceProto);
    }

    /**
     * Reads an instance of {@link SequenceDatasetProto} from the supplied path and deserializes it.
     * @param path The path to read.
     * @return The deserialized sequence dataset.
     * @throws IOException If the path could not be read from, or the parsing failed.
     */
    public static SequenceDataset<?> deserializeFromFile(Path path) throws IOException {
        try (InputStream is = new BufferedInputStream(Files.newInputStream(path))) {
            return deserializeFromStream(is);
        }
    }

    /**
     * Reads an instance of {@link SequenceDatasetProto} from the supplied input stream and deserializes it.
     * @param is The input stream to read.
     * @return The deserialized sequence dataset.
     * @throws IOException If the stream could not be read from, or the parsing failed.
     */
    public static SequenceDataset<?> deserializeFromStream(InputStream is) throws IOException {
        SequenceDatasetProto proto = SequenceDatasetProto.parseFrom(is);
        return deserialize(proto);
    }

    /**
     * Serializes this sequence dataset to a {@link SequenceDatasetProto} and writes it to the supplied path.
     * @param path The path to write to.
     * @throws IOException If the path could not be written to.
     */
    public void serializeToFile(Path path) throws IOException {
        try (OutputStream os = new BufferedOutputStream(Files.newOutputStream(path))) {
            serializeToStream(os);
        }
    }

    /**
     * Serializes this sequence dataset to a {@link SequenceDatasetProto} and writes it to the supplied output stream.
     * <p>
     * Does not close the stream.
     * @param stream The output stream to write to.
     * @throws IOException If the stream could not be written to.
     */
    public void serializeToStream(OutputStream stream) throws IOException {
        SequenceDatasetProto proto = serialize();
        proto.writeTo(stream);
    }

    /**
     * Constructs the data carrier for serialization.
     * @param featureMap The feature domain.
     * @param outputInfo The output domain.
     * @return The serialization data carrier.
     */
    protected DatasetDataCarrier<T> createDataCarrier(FeatureMap featureMap, OutputInfo<T> outputInfo) {
        String version = tribuoVersion == null ? Tribuo.VERSION : tribuoVersion;
        return new DatasetDataCarrier<>(sourceProvenance,featureMap,outputInfo,outputFactory,Collections.emptyList(),version);
    }

    private static class FlatDataset<T extends Output<T>> extends ImmutableDataset<T> {
        private static final long serialVersionUID = 1L;

        FlatDataset(SequenceDataset<T> sequenceDataset) {
            super(sequenceDataset.sourceProvenance, sequenceDataset.outputFactory, sequenceDataset.getFeatureIDMap(), sequenceDataset.getOutputIDInfo());
            for (SequenceExample<T> seq : sequenceDataset) {
                for (Example<T> e : seq) {
                    data.add(e);
                }
            }
        }
    }

    /**
     * Deserializes a list of sequence example protos into a list of sequence examples.
     * @param examplesList The protos.
     * @param outputClass The output class.
     * @param fmap The feature domain.
     * @return The list of deserialized sequence examples.
     */
    protected static List<SequenceExample<?>> deserializeExamples(List<SequenceExampleProto> examplesList, Class<?> outputClass, FeatureMap fmap) {
        List<SequenceExample<?>> examples = new ArrayList<>();
        for (SequenceExampleProto e : examplesList) {
            SequenceExample<?> seq = SequenceExample.deserialize(e);
            for (Example<?> example : seq) {
                if (example.getOutput().getClass().equals(outputClass)) {
                    for (Feature f : example) {
                        if (fmap.get(f.getName()) == null) {
                            throw new IllegalStateException("Invalid protobuf, feature domain does not contain feature " + f.getName() + " present in an example");
                        }
                    }
                } else {
                    throw new IllegalStateException("Invalid protobuf, expected all examples to have output class " + outputClass + ", but found " + example.getOutput().getClass());
                }
            }
            examples.add(seq);
        }
        return examples;
    }
}

