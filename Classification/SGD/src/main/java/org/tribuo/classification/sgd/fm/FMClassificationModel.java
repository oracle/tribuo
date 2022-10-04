/*
 * Copyright (c) 2021, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.sgd.fm;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.ONNXExportable;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.sgd.protos.FMClassificationModelProto;
import org.tribuo.common.sgd.AbstractFMModel;
import org.tribuo.common.sgd.FMParameters;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.util.VectorNormalizer;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.util.onnx.ONNXNode;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The inference time version of a factorization machine trained using SGD.
 * <p>
 * See:
 * <pre>
 * Rendle, S.
 * Factorization machines.
 * 2010 IEEE International Conference on Data Mining
 * </pre>
 */
public class FMClassificationModel extends AbstractFMModel<Label> implements ONNXExportable {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final VectorNormalizer normalizer;

    /**
     * Constructs a classification factorization machine trained via SGD.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param parameters The model parameters.
     * @param normalizer The normalization function.
     * @param generatesProbabilities Does this model generate probabilities?
     */
    FMClassificationModel(String name, ModelProvenance provenance,
                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> outputIDInfo,
                          FMParameters parameters, VectorNormalizer normalizer, boolean generatesProbabilities) {
        super(name, provenance, featureIDMap, outputIDInfo, parameters, generatesProbabilities);
        this.normalizer = normalizer;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static FMClassificationModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        FMClassificationModelProto proto = message.unpack(FMClassificationModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Label.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a label domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Label> outputDomain = (ImmutableOutputInfo<Label>) carrier.outputDomain();

        Parameters params = Parameters.deserialize(proto.getParams());
        if (!(params instanceof FMParameters)) {
            throw new IllegalStateException("Invalid protobuf, parameters must be FMParameters, found " + params.getClass());
        }

        VectorNormalizer normalizer = VectorNormalizer.deserialize(proto.getNormalizer());

        return new FMClassificationModel(carrier.name(),carrier.provenance(),carrier.featureDomain(),outputDomain,(FMParameters) params, normalizer, carrier.generatesProbabilities());
    }

    @Override
    public Prediction<Label> predict(Example<Label> example) {
        PredAndActive predTuple = predictSingle(example);
        DenseVector prediction = predTuple.prediction;
        prediction.normalize(normalizer);

        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        Map<String,Label> predMap = new LinkedHashMap<>();
        for (int i = 0; i < prediction.size(); i++) {
            String labelName = outputIDInfo.getOutput(i).getLabel();
            double score = prediction.get(i);
            Label label = new Label(labelName, score);
            predMap.put(labelName,label);
            if (score > maxScore) {
                maxScore = score;
                maxLabel = label;
            }
        }
        return new Prediction<>(maxLabel, predMap, predTuple.numActiveFeatures, example, generatesProbabilities);
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<Label> carrier = createDataCarrier();
        FMClassificationModelProto.Builder modelBuilder = FMClassificationModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setParams(modelParameters.serialize());
        modelBuilder.setNormalizer(normalizer.serialize());

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(FMClassificationModel.class.getName());
        builder.setSerializedData(Any.pack(modelBuilder.build()));

        return builder.build();
    }

    @Override
    protected FMClassificationModel copy(String newName, ModelProvenance newProvenance) {
        return new FMClassificationModel(newName,newProvenance,featureIDMap,outputIDInfo,(FMParameters)modelParameters.copy(),normalizer,generatesProbabilities);
    }

    @Override
    protected String getDimensionName(int index) {
        return outputIDInfo.getOutput(index).getLabel();
    }

    @Override
    protected String onnxModelName() {
        return "FMClassificationModel";
    }

    @Override
    protected ONNXNode onnxOutput(ONNXNode input) {
        return normalizer.exportNormalizer(input);
    }

}
