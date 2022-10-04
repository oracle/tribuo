/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.multilabel.baseline;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.FeatureMap;
import org.tribuo.Output;
import org.tribuo.classification.Label;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.protos.BinaryExampleProto;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.ExampleProto;
import org.tribuo.util.Merger;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Logger;

/**
 * A simple example which wraps a MultiLabel example, converting the
 * MultiLabel to the presence or absence of a single Label.
 */
class BinaryExample extends Example<Label> {
    private static final Logger logger = Logger.getLogger(BinaryExample.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final Example<MultiLabel> innerExample;

    private Label binaryLabel;

    private final ArrayList<Feature> additionalFeatures = new ArrayList<>();

    /**
     * Creates a BinaryExample, which wraps a MultiLabel example and
     * has a single Label inside.
     * @param innerExample The example to wrap.
     * @param newLabel The new Label.
     */
    BinaryExample(Example<MultiLabel> innerExample, Label newLabel) {
        super(newLabel);
        this.innerExample = innerExample;
        this.binaryLabel = newLabel;
    }

    /**
     * Creates a BinaryExample, which wraps a MultiLabel example and
     * has a single Label inside.
     * @param innerExample The example to wrap.
     * @param newLabel The new Label.
     * @param weight The example weight.
     * @param additionalFeatures The extra features.
     */
    private BinaryExample(Example<MultiLabel> innerExample, Label newLabel, float weight, List<Feature> additionalFeatures) {
        super(newLabel);
        this.weight = weight;
        this.innerExample = innerExample;
        this.binaryLabel = newLabel;
        this.additionalFeatures.addAll(additionalFeatures);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    @SuppressWarnings("unchecked") // guarded by an instanceof check on the output type.
    public static BinaryExample deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        BinaryExampleProto proto = message.unpack(BinaryExampleProto.class);
        Example<?> innerExample = ProtoUtil.deserialize(proto.getInnerExample());
        if (!(innerExample.getOutput() instanceof MultiLabel)) {
            throw new IllegalStateException("Invalid protobuf, inner example did not contain a MultiLabel.");
        }
        Output<?> output = ProtoUtil.deserialize(proto.getLabel());
        if (!(output instanceof Label)) {
            throw new IllegalStateException("Invalid protobuf, binary label is not a Label");
        }
        List<String> names = proto.getFeatureNameList();
        List<Double> values = proto.getFeatureValueList();
        if (names.size() != values.size()) {
            throw new IllegalStateException("Invalid protobuf, names and values don't match. names.size() = " + names.size() + ", values.size() = " + values.size());
        }
        List<Feature> features = new ArrayList<>(names.size());
        for (int i = 0; i < names.size(); i++) {
            features.add(new Feature(names.get(i),values.get(i)));
        }
        return new BinaryExample((Example<MultiLabel>)innerExample,(Label) output,proto.getWeight(),features);
    }

    @Override
    public Label getOutput() {
        return binaryLabel;
    }

    /**
     * Sets a new label.
     * @param newLabel The new label.
     */
    void setLabel(Label newLabel) {
        binaryLabel = newLabel;
    }

    @Override
    protected void sort() {
        logger.warning("Attempting to sort an immutable BinaryExample");
    }

    @Override
    public void add(Feature feature) {
        additionalFeatures.add(feature);
    }

    @Override
    public void addAll(Collection<? extends Feature> features) {
        additionalFeatures.addAll(features);
    }

    @Override
    public int size() {
        return innerExample.size() + additionalFeatures.size();
    }

    @Override
    public void removeFeatures(List<Feature> featureList) {
        logger.warning("Attempting to remove features from an immutable BinaryExample");
    }

    @Override
    public void reduceByName(Merger merger) {
        logger.warning("Attempting to reduce features in an immutable BinaryExample");
    }

    @Override
    public boolean validateExample() {
        return innerExample.validateExample();
    }

    @Override
    public boolean isDense(FeatureMap fMap) {
        return innerExample.isDense(fMap);
    }

    @Override
    protected void densify(List<String> featureNames) {
        logger.warning("Attempting to densify an immutable BinaryExample");
    }

    @Override
    public Example<Label> copy() {
        return new BinaryExample(innerExample, binaryLabel, weight, additionalFeatures);
    }

    @Override
    public void set(Feature feature) {
        logger.warning("Attempting to mutate a feature to an immutable BinaryExample");
    }

    @Override
    public synchronized void setMetadataValue(String key, Object value) {
        logger.warning("Attempting to add metadata to an immutable BinaryExample");
    }

    @Override
    public Iterator<Feature> iterator() {
        List<Feature> tmpFeatures = new ArrayList<>(innerExample.size());
        for (Feature f : innerExample) {
            tmpFeatures.add(f);
        }
        tmpFeatures.addAll(additionalFeatures);
        tmpFeatures.sort(Feature.featureNameComparator());
        return tmpFeatures.iterator();
    }

    @Override
    public void canonicalize(FeatureMap featureMap) {
        logger.finer("Canonicalize is a no-op on BinaryExample.");
    }

    @Override
    public ExampleProto serialize() {
        ExampleProto.Builder builder = ExampleProto.newBuilder();

        builder.setClassName(BinaryExample.class.getName());
        builder.setVersion(CURRENT_VERSION);
        BinaryExampleProto.Builder exampleBuilder = BinaryExampleProto.newBuilder();
        exampleBuilder.setInnerExample(innerExample.serialize());
        exampleBuilder.setWeight(weight);
        exampleBuilder.setLabel(binaryLabel.serialize());
        for (Feature f : additionalFeatures) {
            exampleBuilder.addFeatureName(f.getName());
            exampleBuilder.addFeatureValue(f.getValue());
        }

        builder.setSerializedData(Any.pack(exampleBuilder.build()));

        return builder.build();
    }
}
