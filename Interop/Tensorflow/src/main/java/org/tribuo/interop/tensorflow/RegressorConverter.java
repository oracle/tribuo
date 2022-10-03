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

package org.tribuo.interop.tensorflow;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tensorflow.Operand;
import org.tensorflow.framework.losses.MeanSquaredError;
import org.tensorflow.framework.losses.Reduction;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.index.Indices;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.family.TNumber;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.interop.tensorflow.protos.OutputConverterProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.regression.ImmutableRegressionInfo;
import org.tribuo.regression.Regressor;
import org.tensorflow.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

/**
 * Can convert a {@link Regressor} to a {@link TFloat32} vector and a {@link TFloat32} into a
 * {@link Prediction} or {@code Regressor}.
 */
@ProtoSerializableClass(version = RegressorConverter.CURRENT_VERSION)
public class RegressorConverter implements OutputConverter<Regressor> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Constructs a RegressorConverter.
     */
    public RegressorConverter() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static RegressorConverter deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new RegressorConverter();
    }

    @Override
    public OutputConverterProto serialize() {
        return ProtoUtil.serialize(this);
    }

    /**
     * Returns a mean squared error loss.
     * @return The mse loss.
     */
    @Override
    public BiFunction<Ops, Pair<Placeholder<? extends TNumber>,Operand<TNumber>>,Operand<TNumber>> loss() {
        return (ops, pair) -> new MeanSquaredError("tribuo-mse", Reduction.SUM_OVER_BATCH_SIZE).call(ops,pair.getA(),pair.getB());
    }

    /**
     * Applies the identity function
     * @param <U> The input type (should be TFloat32).
     * @return A function which applies the identity function.
     */
    @Override
    public <U extends TNumber> BiFunction<Ops, Operand<U>, Op> outputTransformFunction() {
        return Ops::identity;
    }

    @Override
    public Prediction<Regressor> convertToPrediction(Tensor tensor, ImmutableOutputInfo<Regressor> outputIDInfo, int numValidFeatures, Example<Regressor> example) {
        Regressor r = convertToOutput(tensor,outputIDInfo);
        return new Prediction<>(r,numValidFeatures,example);
    }

    @Override
    public Regressor convertToOutput(Tensor tensor, ImmutableOutputInfo<Regressor> outputIDInfo) {
        FloatNdArray predictions = getBatchPredictions(tensor, outputIDInfo.size());
        long[] shape = predictions.shape().asArray();
        if (shape[0] != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many results, found " + shape[0]);
        } else if (shape[1] != outputIDInfo.size()) {
            throw new IllegalArgumentException("Supplied tensor has an incorrect number of dimensions, shape[1] = " + shape[1] + ", expected " + outputIDInfo.size());
        }
        String[] names = new String[outputIDInfo.size()];
        double[] values = new double[outputIDInfo.size()];
        // Note this inserts in an ordering which is not necessarily the natural one,
        // but the Regressor constructor sorts it to maintain the natural ordering.
        // The names and the values still line up, so this code is valid.
        for (Pair<Integer,Regressor> p : outputIDInfo) {
            int id = p.getA();
            names[id] = p.getB().getNames()[0];
            values[id] = predictions.getFloat(0,id);
        }
        return new Regressor(names,values);
    }

    /**
     * Converts a tensor into a 2d {@link FloatNdArray}.
     * <p>
     * It coerces a 1d array into a 2d array by adding a new dimension.
     * <p>
     * If the number of output dimensions in the tensor is not correct, or
     * it's not a TFloat32, or it's not 1d or 2d then it throws {@link IllegalArgumentException}.
     * @param tensor The tensor to convert.
     * @param outputDims The number of output regression dimensions.
     * @return A 2d FloatNdArray.
     */
    private FloatNdArray getBatchPredictions(Tensor tensor, int outputDims) {
        if (tensor instanceof TFloat32) {
            long[] shape = tensor.shape().asArray();
            if ((shape.length != 2) && (shape.length != 1)) {
                throw new IllegalArgumentException("Supplied tensor has the wrong number of dimensions, shape = " + Arrays.toString(shape));
            }
            if (shape.length == 1) {
                TFloat32 floatTensor = (TFloat32) tensor;
                // Make a new dimension to make it easier to process in downstream code
                return floatTensor.slice(Indices.all(),Indices.newAxis());
            } else {
                if (shape[1] != outputDims) {
                    throw new IllegalArgumentException("Supplied tensor has incorrect number of elements, tensor value dimension: " + Arrays.toString(shape) + ", output dimension: " + outputDims);
                }
                // No reshaping necessary
                return (TFloat32) tensor;
            }
        } else {
            throw new IllegalArgumentException("Tensor is not a 32-bit float. Found type " + tensor.getClass().getName());
        }
    }

    @Override
    public List<Prediction<Regressor>> convertToBatchPrediction(Tensor tensor, ImmutableOutputInfo<Regressor> outputIDInfo, int[] numValidFeatures, List<Example<Regressor>> examples) {
        List<Regressor> regressors = convertToBatchOutput(tensor,outputIDInfo);
        List<Prediction<Regressor>> output = new ArrayList<>();

        if ((regressors.size() != examples.size()) || (regressors.size() != numValidFeatures.length)) {
            throw new IllegalArgumentException("Invalid number of predictions received from Tensorflow, expected " + numValidFeatures.length + ", received " + regressors.size());
        }

        for (int i = 0; i < regressors.size(); i++) {
            output.add(new Prediction<>(regressors.get(i),numValidFeatures[i],examples.get(i)));
        }

        return output;
    }

    @Override
    public List<Regressor> convertToBatchOutput(Tensor tensor, ImmutableOutputInfo<Regressor> outputIDInfo) {
        FloatNdArray predictions = getBatchPredictions(tensor, outputIDInfo.size());
        List<Regressor> output = new ArrayList<>();
        int batchSize = (int) predictions.shape().asArray()[0];

        // Similar to convertToOutput, names and values are ordered by
        // the id, not the natural ordering, but the Regressor constructor
        // fixes that.
        String[] names = new String[outputIDInfo.size()];
        for (Pair<Integer,Regressor> p : outputIDInfo) {
            int id = p.getA();
            names[id] = p.getB().getNames()[0];
        }
        for (int i = 0; i < batchSize; i++) {
            double[] values = new double[names.length];
            for (int j = 0; j < names.length; j++) {
                values[j] = predictions.getFloat(i,j);
            }
            output.add(new Regressor(names,values));
        }

        return output;
    }

    @Override
    public Tensor convertToTensor(Regressor example, ImmutableOutputInfo<Regressor> outputIDInfo) {
        TFloat32 output = TFloat32.tensorOf(Shape.of(1,outputIDInfo.size()));
        // We map through the id to natural order mapping as regressor might not
        // be stored in the id order.
        int[] ids = ((ImmutableRegressionInfo) outputIDInfo).getIDtoNaturalOrderMapping();
        double[] values = example.getValues();
        for (Pair<Integer,Regressor> p : outputIDInfo) {
            int id = p.getA();
            output.setFloat((float) values[ids[id]],0,id);
        }
        return output;
    }

    @Override
    public Tensor convertToTensor(List<Example<Regressor>> examples, ImmutableOutputInfo<Regressor> outputIDInfo) {
        TFloat32 output = TFloat32.tensorOf(Shape.of(examples.size(),outputIDInfo.size()));
        // We map through the id to natural order mapping as regressor might not
        // be stored in the id order.
        int[] ids = ((ImmutableRegressionInfo) outputIDInfo).getIDtoNaturalOrderMapping();
        int i = 0;
        for (Example<Regressor> e : examples) {
            double[] values = e.getOutput().getValues();
            for (int j = 0; j < outputIDInfo.size(); j++) {
                output.setFloat((float)values[ids[j]],i,j);
            }
            i++;
        }
        return output;
    }

    @Override
    public boolean generatesProbabilities() {
        return false;
    }

    @Override
    public String toString() {
        return "RegressorConverter()";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"OutputConverter");
    }

    @Override
    public Class<Regressor> getTypeWitness() {
        return Regressor.class;
    }
}
