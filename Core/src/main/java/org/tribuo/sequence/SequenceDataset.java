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
import org.tribuo.protos.core.SequenceDatasetProto;
import org.tribuo.protos.core.SequenceExampleProto;
import org.tribuo.provenance.DataProvenance;
import org.tribuo.provenance.DatasetProvenance;

import java.io.Serializable;
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

    protected SequenceDataset(DataProvenance sourceProvenance, OutputFactory<T> outputFactory) {
        this(sourceProvenance, outputFactory, Tribuo.VERSION);
    }

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

        public FlatDataset(SequenceDataset<T> sequenceDataset) {
            super(sequenceDataset.sourceProvenance, sequenceDataset.outputFactory, sequenceDataset.getFeatureIDMap(), sequenceDataset.getOutputIDInfo());
            for (SequenceExample<T> seq : sequenceDataset) {
                for (Example<T> e : seq) {
                    data.add(e);
                }
            }
        }
    }

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

