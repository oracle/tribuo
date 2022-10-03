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

package org.tribuo.classification.baseline;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.classification.ImmutableLabelInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.baseline.DummyClassifierTrainer.DummyType;
import org.tribuo.classification.protos.DummyClassifierModelProto;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.util.Util;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.stream.Collectors;

import static org.tribuo.Trainer.DEFAULT_SEED;

/**
 * A model which performs dummy classifications (e.g., constant output, uniform sampled labels, stratified sampled labels).
 */
public class DummyClassifierModel extends Model<Label> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final DummyType dummyType;

    private final Label constantLabel;

    private final double[] cdf;

    private final Random rng;

    private final long seed;

    DummyClassifierModel(ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> outputIDInfo) {
        super("dummy-MOST_FREQUENT-classifier", description, featureIDMap, outputIDInfo, false);
        this.dummyType = DummyType.MOST_FREQUENT;
        this.constantLabel = findMostFrequentLabel(outputIDInfo);
        this.cdf = null;
        this.seed = DEFAULT_SEED;
        this.rng = null;
    }

    DummyClassifierModel(ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> outputIDInfo, DummyType dummyType, long seed) {
        super("dummy-"+dummyType+"-classifier", description, featureIDMap, outputIDInfo, false);
        this.dummyType = dummyType;
        this.constantLabel = LabelFactory.UNKNOWN_LABEL;
        this.cdf = dummyType == DummyType.UNIFORM ? generateUniformCDF(outputIDInfo) : generateStratifiedCDF(outputIDInfo);
        this.seed = seed;
        this.rng = new Random(seed);
    }

    DummyClassifierModel(ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> outputIDInfo, Label constantLabel) {
        super("dummy-CONSTANT-classifier", description, featureIDMap, outputIDInfo, false);
        this.dummyType = DummyType.CONSTANT;
        this.constantLabel = constantLabel;
        this.cdf = null;
        this.seed = DEFAULT_SEED;
        this.rng = null;
    }

    private DummyClassifierModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureMap,
                                 ImmutableOutputInfo<Label> outputInfo, DummyType type, Label constantLabel,
                                 double[] cdf, long seed) {
        super(name, provenance, featureMap, outputInfo, false);
        this.dummyType = type;
        this.constantLabel = constantLabel;
        this.cdf = cdf;
        this.seed = seed;
        this.rng = new Random(seed);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static DummyClassifierModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        DummyClassifierModelProto proto = message.unpack(DummyClassifierModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Label.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a label domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Label> outputDomain = (ImmutableOutputInfo<Label>) carrier.outputDomain();

        DummyType dummyType = DummyType.valueOf(proto.getDummyType());

        Output<?> output = Output.deserialize(proto.getConstantLabel());
        if (!(output instanceof Label)) {
            throw new IllegalStateException("Invalid protobuf, expected a label, found " + output.getClass());
        }
        Label constantLabel = (Label) output;

        double[] cdf = null;
        if (proto.getCdfCount() > 0) {
            cdf = Util.toPrimitiveDouble(proto.getCdfList());
        }

        long seed = proto.getSeed();

        return new DummyClassifierModel(carrier.name(), carrier.provenance(), carrier.featureDomain(), outputDomain,
            dummyType, constantLabel, cdf, seed);
    }

    @Override
    public Prediction<Label> predict(Example<Label> example) {
        switch (dummyType) {
            case CONSTANT:
            case MOST_FREQUENT:
                return new Prediction<>(constantLabel,0,example);
            case UNIFORM:
            case STRATIFIED:
                return new Prediction<>(sampleLabel(cdf,outputIDInfo,rng),0,example);
            default:
                throw new IllegalStateException("Unknown dummyType " + dummyType);
        }
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        Map<String,List<Pair<String,Double>>> map = new HashMap<>();
        if (n != 0) {
            map.put(Model.ALL_OUTPUTS, Collections.singletonList(new Pair<>(BIAS_FEATURE, 1.0)));
        }
        return map;
    }

    @Override
    public Optional<Excuse<Label>> getExcuse(Example<Label> example) {
        return Optional.of(new Excuse<>(example,predict(example),getTopFeatures(1)));
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<Label> carrier = createDataCarrier();

        DummyClassifierModelProto.Builder modelBuilder = DummyClassifierModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setDummyType(dummyType.name());
        modelBuilder.setConstantLabel(constantLabel.serialize());
        if (cdf != null) {
            modelBuilder.addAllCdf(Arrays.stream(cdf).boxed().collect(Collectors.toList()));
        }
        modelBuilder.setSeed(seed);

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(DummyClassifierModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    @Override
    protected DummyClassifierModel copy(String newName, ModelProvenance newProvenance) {
        switch (dummyType) {
            case CONSTANT:
                return new DummyClassifierModel(newProvenance,featureIDMap,outputIDInfo,constantLabel.copy());
            case MOST_FREQUENT:
                return new DummyClassifierModel(newProvenance,featureIDMap,outputIDInfo);
            case UNIFORM:
            case STRATIFIED:
                return new DummyClassifierModel(newProvenance,featureIDMap,outputIDInfo,dummyType,seed);
            default:
                throw new IllegalStateException("Unknown dummyType " + dummyType);
        }
    }

    /**
     * Samples a label from the supplied CDF.
     * @param cdf The CDF to sample from.
     * @param outputIDInfo The mapping from label ids to values.
     * @param rng The RNG to use.
     * @return A Label.
     */
    private static Label sampleLabel(double[] cdf, ImmutableOutputInfo<Label> outputIDInfo, Random rng) {
        int sample = Util.sampleFromCDF(cdf,rng);
        return outputIDInfo.getOutput(sample);
    }

    /**
     * Finds the most frequent label and returns it.
     * @param outputInfo The output information (must be a subclass of ImmutableLabelInfo).
     * @return The most frequent label.
     */
    private static Label findMostFrequentLabel(ImmutableOutputInfo<Label> outputInfo) {
        Label maxLabel = null;
        long count = -1;

        ImmutableLabelInfo labelInfo = (ImmutableLabelInfo) outputInfo;

        for (Pair<Integer,Label> p : labelInfo) {
            long curCount = labelInfo.getLabelCount(p.getA());
            if (curCount > count) {
                count = curCount;
                maxLabel = p.getB();
            }
        }

        return maxLabel;
    }

    /**
     * Generates a uniform CDF for the supplied labels.
     * @param outputInfo The output information.
     * @return A uniform CDF across the domain.
     */
    private static double[] generateUniformCDF(ImmutableOutputInfo<Label> outputInfo) {
        int length = outputInfo.getDomain().size();
        double[] pmf = Util.generateUniformVector(length,1.0/length);
        return Util.generateCDF(pmf);
    }

    /**
     * Generates a CDF where the label probabilities are proportional to their observed counts.
     * @param outputInfo The output information.
     * @return A CDF proportional to the observed counts.
     */
    private static double[] generateStratifiedCDF(ImmutableOutputInfo<Label> outputInfo) {
        ImmutableLabelInfo labelInfo = (ImmutableLabelInfo) outputInfo;
        int length = labelInfo.getDomain().size();
        long counts = labelInfo.getTotalObservations();

        double[] pmf = new double[length];

        for (Pair<Integer,Label> p : labelInfo) {
            int idx = p.getA();
            pmf[idx] = labelInfo.getLabelCount(idx) / (double) counts;
        }

        return Util.generateCDF(pmf);
    }
}
