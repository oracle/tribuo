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

package org.tribuo.regression.baseline;

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
import org.tribuo.Trainer;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.baseline.DummyRegressionTrainer.DummyType;
import org.tribuo.regression.protos.DummyRegressionModelProto;
import org.tribuo.util.Util;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * A model which performs dummy regressions (e.g., constant output, gaussian sampled output, mean value, median, quartile).
 */
public class DummyRegressionModel extends Model<Regressor> {
    private static final long serialVersionUID = 2L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final DummyType dummyType;

    private final Regressor output;

    private final long seed;

    private final Random rng;

    private final double[] means;

    private final double[] variances;

    private final String[] dimensionNames;

    DummyRegressionModel(ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo, long seed, double[] means, double[] variances, String[] names) {
        super("dummy-GAUSSIAN-regression", description, featureIDMap, outputIDInfo, false);
        this.dummyType = DummyType.GAUSSIAN;
        this.output = null;
        this.seed = seed;
        this.rng = new Random(seed);
        this.means = Arrays.copyOf(means,means.length);
        this.variances = Arrays.copyOf(variances,variances.length);
        this.dimensionNames = Arrays.copyOf(names,names.length);
    }

    DummyRegressionModel(ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo, DummyType dummyType, Regressor regressor) {
        super("dummy-"+dummyType+"-regression", description, featureIDMap, outputIDInfo, false);
        this.dummyType = dummyType;
        this.output = regressor;
        this.seed = Trainer.DEFAULT_SEED;
        this.rng = null;
        this.means = new double[0];
        this.variances = new double[0];
        this.dimensionNames = new String[0];
    }

    private DummyRegressionModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureMap,
                                 ImmutableOutputInfo<Regressor> outputInfo, DummyType dummyType, Regressor regressor,
                                 long seed, double[] means, double[] variances, String[] dimensionNames) {
        super(name, provenance, featureMap, outputInfo, false);
        this.dummyType = dummyType;
        this.output = regressor;
        this.seed = seed;
        this.rng = new Random(seed);
        this.means = means;
        this.variances = variances;
        this.dimensionNames = dimensionNames;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static DummyRegressionModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        DummyRegressionModelProto proto = message.unpack(DummyRegressionModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Regressor.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a regression domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Regressor> outputDomain = (ImmutableOutputInfo<Regressor>) carrier.outputDomain();

        DummyType dummyType = DummyType.valueOf(proto.getDummyType());

        Regressor constantRegressor = null;
        if (!dummyType.equals(DummyType.GAUSSIAN)) {
            Output<?> output = Output.deserialize(proto.getOutput());
            if (!(output instanceof Regressor)) {
                throw new IllegalStateException("Invalid protobuf, expected a Regressor, found " + output.getClass());
            }
            constantRegressor = (Regressor) output;
        }

        long seed = proto.getSeed();

        double[] means = Util.toPrimitiveDouble(proto.getMeansList());
        double[] variances = Util.toPrimitiveDouble(proto.getVariancesList());
        String[] dimensionNames = proto.getDimensionNamesList().toArray(new String[0]);

        return new DummyRegressionModel(carrier.name(), carrier.provenance(), carrier.featureDomain(), outputDomain,
            dummyType, constantRegressor, seed, means, variances, dimensionNames);
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        switch (dummyType) {
            case CONSTANT:
            case MEAN:
            case MEDIAN:
            case QUARTILE:
                return new Prediction<>(output,0,example);
            case GAUSSIAN: {
                Regressor.DimensionTuple[] dimensions = new Regressor.DimensionTuple[dimensionNames.length];
                for (int i = 0; i < dimensionNames.length; i++) {
                    double regressionValue = (rng.nextGaussian() * variances[i]) + means[i];
                    dimensions[i] = new Regressor.DimensionTuple(dimensionNames[i],regressionValue);
                }
                return new Prediction<>(new Regressor(dimensions), 0, example);
            }
            default:
                throw new IllegalStateException("Unknown dummyType " + dummyType);
        }
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        if (n != 0) {
            return Collections.singletonMap(Model.ALL_OUTPUTS, Collections.singletonList(new Pair<>(BIAS_FEATURE, 1.0)));
        } else {
            return Collections.emptyMap();
        }
    }

    @Override
    public Optional<Excuse<Regressor>> getExcuse(Example<Regressor> example) {
        return Optional.of(new Excuse<>(example,predict(example),getTopFeatures(1)));
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<Regressor> carrier = createDataCarrier();

        DummyRegressionModelProto.Builder modelBuilder = DummyRegressionModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setDummyType(dummyType.name());
        if (output != null) {
            modelBuilder.setOutput(output.serialize());
        }
        modelBuilder.addAllMeans(Arrays.stream(means).boxed().collect(Collectors.toList()));
        modelBuilder.addAllVariances(Arrays.stream(variances).boxed().collect(Collectors.toList()));
        modelBuilder.addAllDimensionNames(Arrays.asList(dimensionNames));
        modelBuilder.setSeed(seed);

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(DummyRegressionModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    @Override
    protected Model<Regressor> copy(String newName, ModelProvenance newProvenance) {
        switch (dummyType) {
            case GAUSSIAN:
                return new DummyRegressionModel(newProvenance,featureIDMap,outputIDInfo,seed,means,variances,dimensionNames);
            case CONSTANT:
            case MEAN:
            case MEDIAN:
            case QUARTILE:
                return new DummyRegressionModel(newProvenance,featureIDMap,outputIDInfo,dummyType,output.copy());
            default:
                throw new IllegalStateException("Unknown dummyType " + dummyType);
        }
    }
}
