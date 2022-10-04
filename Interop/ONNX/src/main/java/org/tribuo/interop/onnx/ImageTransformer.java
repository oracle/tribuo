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

package org.tribuo.interop.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.interop.onnx.protos.ExampleTransformerProto;
import org.tribuo.interop.onnx.protos.ImageTransformerProto;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Objects;

/**
 * Image transformer. Assumes the feature id numbers are linearised ids of the form
 * [0,0,0] = 0, [1,0,0] = 1, ..., [i,0,0] = i, [0,1,0] = i+1, ..., [i,j,0] = i*j, ...
 * [0,0,1] = (i*j)+1, ..., [i,j,k] = i*j*k.
 * <p>
 * ONNX expects images in the format [channels,height,width].
 */
@ProtoSerializableClass(serializedDataClass = ImageTransformerProto.class, version = ImageTransformer.CURRENT_VERSION)
public class ImageTransformer implements ExampleTransformer {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    @Config(mandatory=true,description="Image width.")
    @ProtoSerializableField
    private int width;

    @Config(mandatory=true,description="Image height.")
    @ProtoSerializableField
    private int height;

    @Config(mandatory=true,description="Number of channels.")
    @ProtoSerializableField
    private int channels;

    /**
     * For olcut.
     */
    private ImageTransformer() {}

    /**
     * Constructs an image transformer with the specified parameters.
     * @param channels The number of colour channels.
     * @param height The height.
     * @param width The width.
     */
    public ImageTransformer(int channels, int height, int width) {
        if (width < 1 || height < 1 || channels < 1) {
            throw new PropertyException("","Inputs must be positive integers, found [c="+channels+",h="+height+",w="+width+"]");
        }
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
            throw new PropertyException("","Inputs must be positive integers, found [c="+channels+",h="+height+",w="+width+"]");
        }
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static ImageTransformer deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        ImageTransformerProto proto = message.unpack(ImageTransformerProto.class);
        return new ImageTransformer(proto.getChannels(), proto.getHeight(), proto.getWidth());
    }

    /**
     * Actually performs the transformation. Pads unseen values
     * with zero. Writes to the buffer in multidimensional row-major form.
     * @param buffer The buffer to write to.
     * @param startPos The starting position of the buffer.
     * @param vector The vector to transform.
     */
    private void innerTransform(FloatBuffer buffer, int startPos, SparseVector vector) {
        for (VectorTuple f : vector) {
            int id = f.index;
            buffer.put(id+startPos,(float)f.value);
        }
    }

    @Override
    public OnnxTensor transform(OrtEnvironment env, SparseVector vector) throws OrtException {
        FloatBuffer buffer = ByteBuffer.allocateDirect(vector.size()*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        innerTransform(buffer,0,vector);
        buffer.rewind(); // rewind the buffer as createTensor now reads from the current position.
        return OnnxTensor.createTensor(env,buffer,new long[]{1,channels,height,width});
    }

    @Override
    public OnnxTensor transform(OrtEnvironment env, List<SparseVector> vectors) throws OrtException {
        if (vectors.isEmpty()) {
            return OnnxTensor.createTensor(env,FloatBuffer.allocate(0),new long[]{0,channels,height,width});
        } else {
            int initialSize = vectors.get(0).size();
            FloatBuffer buffer = ByteBuffer.allocateDirect(initialSize * vectors.size() * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
            int position = 0;
            for (SparseVector v : vectors) {
                innerTransform(buffer, position, v);
                position += v.size();
                if (v.size() != initialSize) {
                    throw new IllegalArgumentException("Vectors are not all the same dimension, expected " + initialSize + ", found " + v.size());
                }
            }
            buffer.rewind(); // rewind the buffer as createTensor now reads from the current position.
            return OnnxTensor.createTensor(env, buffer, new long[]{vectors.size(), channels, height, width});
        }
    }

    @Override
    public String toString() {
        return "ImageTransformer(channels="+channels+",height="+height+",width="+width+")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ImageTransformer that = (ImageTransformer) o;
        return width == that.width && height == that.height && channels == that.channels;
    }

    @Override
    public int hashCode() {
        return Objects.hash(width, height, channels);
    }

    @Override
    public ExampleTransformerProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"ExampleTransformer");
    }
}
