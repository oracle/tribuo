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
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.types.TFloat32;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.interop.tensorflow.protos.FeatureConverterProto;
import org.tribuo.interop.tensorflow.protos.ImageConverterProto;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;

import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Image converter. Assumes the feature id numbers are linearised ids of the form:
 * <pre>
 * [0,0,0] = 0, [0,0,1] = 1, [0,1,0] = k, [1,0,0] = j*k, ..., [i,0,0] = i*j*k,
 * </pre>
 * That is, they are in multidimensional row major order (e.g. the order used by {@link org.tribuo.datasource.IDXDataSource}).
 */
@ProtoSerializableClass(serializedDataClass = ImageConverterProto.class, version = ImageConverter.CURRENT_VERSION)
public class ImageConverter implements FeatureConverter {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    @Config(mandatory=true,description="TensorFlow Placeholder Input name.")
    @ProtoSerializableField
    private String inputName;

    @Config(mandatory=true,description="Image width.")
    @ProtoSerializableField
    private int width;

    @Config(mandatory=true,description="Image height.")
    @ProtoSerializableField
    private int height;

    @Config(mandatory=true,description="Number of channels.")
    @ProtoSerializableField
    private int channels;

    private int totalPixels;

    /**
     * For olcut.
     */
    private ImageConverter() {}

    /**
     * Builds an image converter for images of the supplied size.
     * @param inputName The input name.
     * @param width The image width.
     * @param height The image height.
     * @param channels The number of colour channels.
     */
    public ImageConverter(String inputName, int width, int height, int channels) {
        if (width < 1 || height < 1 || channels < 1) {
            throw new IllegalArgumentException("Inputs must be positive integers, found ["+width+","+height+","+channels+"]");
        }
        if (inputName == null || inputName.isEmpty()) {
            throw new IllegalArgumentException("The input name must be a valid String");
        }
        long values = ((long)width)*height*channels;
        if (values > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Image size must be less than 2^31, found " + values);
        }
        this.inputName = inputName;
        this.totalPixels = (int) values;
        this.width = width;
        this.height = height;
        this.channels = channels;
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (width < 1 || height < 1 || channels < 1) {
            throw new PropertyException("","Inputs must be positive integers, found ["+width+","+height+","+channels+"]");
        }
        long values = ((long)width)*height*channels;
        if (values > Integer.MAX_VALUE) {
            throw new PropertyException("","Image size must be less than 2^31, found " + values);
        }
        this.totalPixels = (int) values;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static ImageConverter deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        ImageConverterProto proto = message.unpack(ImageConverterProto.class);
        return new ImageConverter(proto.getInputName(), proto.getWidth(), proto.getHeight(), proto.getChannels());
    }

    @Override
    public FeatureConverterProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public Set<String> inputNamesSet() {
        return Collections.singleton(inputName);
    }

    /**
     * Transform implicitly pads unseen values with zero.
     * @param example The example to transform.
     * @param featureIDMap The feature id mapping to use.
     * @return A 4d tensor, (1, width, height, channels) for this example.
     */
    @Override
    public TensorMap convert(Example<?> example, ImmutableFeatureMap featureIDMap) {
        float[] image = innerTransform(example,featureIDMap);
        return new TensorMap(inputName,TFloat32.tensorOf(Shape.of(1,width,height,channels), DataBuffers.of(image)));
    }

    /**
     * Actually performs the transformation. Implicitly pads unseen values
     * with zero.
     * @param example The example to transform.
     * @param featureIDMap The feature id mapping to use.
     * @return A 1d array stored in multidimensional column-major order representing the example.
     */
    float[] innerTransform(Example<?> example, ImmutableFeatureMap featureIDMap) {
        if (featureIDMap.size() > totalPixels) {
            throw new IllegalArgumentException("Found more values than expected, expected " + totalPixels + ", found " + featureIDMap.size());
        }

        float[] output = new float[totalPixels];

        for (Feature f : example) {
            int id = featureIDMap.getID(f.getName());
            output[id] = (float) f.getValue();
        }

        return output;
    }

    /**
     * Actually performs the transformation. Implicitly pads unseen values
     * with zero.
     * @param vector The vector to transform.
     * @return A 1d array stored in multidimensional column-major order representing the example.
     */
    float[] innerTransform(SGDVector vector) {
        if (vector.size() > totalPixels) {
            throw new IllegalArgumentException("Found more values than expected, expected " + totalPixels + ", found " + vector.size());
        }
        float[] output = new float[totalPixels];

        for (VectorTuple f : vector) {
            output[f.index] = (float) f.value;
        }

        return output;
    }

    /**
     * Transform implicitly pads unseen values with zero.
     * <p>
     * Converts a batch of examples into a Tensor.
     * @param examples The examples to transform.
     * @param featureIDMap The feature id mapping to use.
     * @return A 4d tensor, (batch-id, width, height, channels) for this example.
     */
    @Override
    public TensorMap convert(List<? extends Example<?>> examples, ImmutableFeatureMap featureIDMap) {
        TFloat32 output = TFloat32.tensorOf(Shape.of(examples.size(),width,height,channels));

        int i = 0;
        for (Example<?> example : examples) {
            float[] features = innerTransform(example,featureIDMap);
            output.set(NdArrays.wrap(Shape.of(width,height,channels), DataBuffers.of(features)),i);
            i++;
        }

        return new TensorMap(inputName,output);
    }

    @Override
    public TensorMap convert(SGDVector vector) {
        float[] image = innerTransform(vector);
        return new TensorMap(inputName,TFloat32.tensorOf(Shape.of(1,width,height,channels), DataBuffers.of(image)));
    }

    @Override
    public TensorMap convert(List<? extends SGDVector> vectors) {
        TFloat32 output = TFloat32.tensorOf(Shape.of(vectors.size(),width,height,channels));

        int i = 0;
        for (SGDVector vector : vectors) {
            float[] features = innerTransform(vector);
            output.set(NdArrays.wrap(Shape.of(width,height,channels), DataBuffers.of(features)),i);
            i++;
        }

        return new TensorMap(inputName,output);
    }

    @Override
    public String toString() {
        return "ImageConverter(inputName='"+inputName+"',width="+width+",height="+height+",channels="+channels+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FeatureConverter");
    }
}
