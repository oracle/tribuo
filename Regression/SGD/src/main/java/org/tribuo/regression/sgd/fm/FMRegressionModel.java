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

package org.tribuo.regression.sgd.fm;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.ONNXExportable;
import org.tribuo.Prediction;
import org.tribuo.common.sgd.AbstractFMModel;
import org.tribuo.common.sgd.FMParameters;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.math.Parameters;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.ImmutableRegressionInfo;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.sgd.protos.FMRegressionModelProto;
import org.tribuo.util.onnx.ONNXInitializer;
import org.tribuo.util.onnx.ONNXNode;
import org.tribuo.util.onnx.ONNXOperators;

import java.util.Arrays;

/**
 * The inference time model of a regression factorization machine trained using SGD.
 * Independently predicts each output dimension, unless they are tied together in the
 * optimiser.
 * <p>
 * See:
 * <pre>
 * Rendle, S.
 * Factorization machines.
 * 2010 IEEE International Conference on Data Mining
 * </pre>
 */
public class FMRegressionModel extends AbstractFMModel<Regressor> implements ONNXExportable {
    private static final long serialVersionUID = 3L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final String[] dimensionNames;

    private final boolean standardise;

    /**
     * Constructs a linear regression model trained via SGD.
     * @param name The model name.
     * @param dimensionNames The regression dimension names.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param parameters The model parameters.
     * @param standardise Is the model fitted on standardised regressors?
     */
    FMRegressionModel(String name, String[] dimensionNames, ModelProvenance provenance,
                      ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo,
                      FMParameters parameters, boolean standardise) {
        super(name, provenance, featureIDMap, outputIDInfo, parameters, false);
        this.dimensionNames = dimensionNames;
        this.standardise = standardise;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static FMRegressionModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        FMRegressionModelProto proto = message.unpack(FMRegressionModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Regressor.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a regression domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Regressor> outputDomain = (ImmutableOutputInfo<Regressor>) carrier.outputDomain();

        Parameters params = Parameters.deserialize(proto.getParams());
        if (!(params instanceof FMParameters)) {
            throw new IllegalStateException("Invalid protobuf, parameters must be FMParameters, found " + params.getClass());
        }

        String[] dimensionNames = proto.getDimensionNamesList().toArray(new String[0]);
        if (dimensionNames.length != outputDomain.size()) {
            throw new IllegalStateException("Invalid protobuf, found a different number of dimension names to the output dimensions, found " + dimensionNames.length + " , expected " + outputDomain.size());
        }

        return new FMRegressionModel(carrier.name(), dimensionNames, carrier.provenance(), carrier.featureDomain(),
                outputDomain, (FMParameters) params, proto.getStandardise());
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        PredAndActive predTuple = predictSingle(example);
        double[] predictions = predTuple.prediction.toArray();
        if (standardise) {
            predictions = unstandardisePredictions(predictions);
        }
        return new Prediction<>(new Regressor(dimensionNames,predictions), predTuple.numActiveFeatures, example);
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<Regressor> carrier = createDataCarrier();
        FMRegressionModelProto.Builder modelBuilder = FMRegressionModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setParams(modelParameters.serialize());
        modelBuilder.addAllDimensionNames(Arrays.asList(dimensionNames));
        modelBuilder.setStandardise(standardise);

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(FMRegressionModel.class.getName());
        builder.setSerializedData(Any.pack(modelBuilder.build()));

        return builder.build();
    }

    /**
     * Converts zero mean unit variance predictions into the true range.
     * @param predictions The predictions to convert.
     */
    private double[] unstandardisePredictions(double[] predictions) {
        ImmutableRegressionInfo info = (ImmutableRegressionInfo) outputIDInfo;
        for (int i = 0; i < predictions.length; i++) {
            double mean = info.getMean(i);
            double variance = info.getVariance(i);
            predictions[i] = (predictions[i] * variance) + mean;
        }
        return predictions;
    }

    @Override
    protected FMRegressionModel copy(String newName, ModelProvenance newProvenance) {
        return new FMRegressionModel(newName,Arrays.copyOf(dimensionNames,dimensionNames.length),newProvenance,featureIDMap,outputIDInfo,(FMParameters)modelParameters.copy(),standardise);
    }

    @Override
    protected String getDimensionName(int index) {
        return dimensionNames[index];
    }

    @Override
    protected String onnxModelName() {
        return "FMRegressionModel";
    }

    @Override
    protected ONNXNode onnxOutput(ONNXNode fmOutput) {
        if(standardise) {
            ImmutableRegressionInfo info = (ImmutableRegressionInfo) outputIDInfo;
            double[] means = new double[outputIDInfo.size()];
            double[] variances = new double[outputIDInfo.size()];
            for (int i = 0; i < means.length; i++) {
                means[i] = info.getMean(i);
                variances[i] = info.getVariance(i);
            }
            ONNXInitializer outputMean = fmOutput.onnxContext().array("y_mean", means);
            ONNXInitializer outputVariance = fmOutput.onnxContext().array("y_var", variances);

            return fmOutput.apply(ONNXOperators.MUL, outputVariance).apply(ONNXOperators.ADD, outputMean);
        } else {
            return fmOutput;
        }
    }
}
