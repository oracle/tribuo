/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math.neighbour.bruteforce;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.tribuo.math.distance.Distance;
import org.tribuo.math.distance.DistanceType;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.neighbour.NeighboursQueryFactory;
import org.tribuo.math.protos.BruteForceFactoryProto;
import org.tribuo.math.protos.NeighbourFactoryProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;

import java.util.Objects;

/**
 * A factory which creates brute-force nearest neighbour query objects.
 */
@ProtoSerializableClass(version = NeighboursBruteForceFactory.CURRENT_VERSION, serializedDataClass = BruteForceFactoryProto.class)
public final class NeighboursBruteForceFactory implements NeighboursQueryFactory {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    @Config(description = "The distance function to use.")
    @ProtoSerializableField
    private Distance distance = DistanceType.L2.getDistance();

    @Config(description = "The number of threads to use for training.")
    @ProtoSerializableField
    private int numThreads = 1;

    /**
     * for olcut.
     */
    private NeighboursBruteForceFactory() {}

    /**
     * Constructs a brute-force nearest neighbor query factory object using the supplied parameters.
     * @param distance The distance function.
     * @param numThreads The number of threads to be used to parallelize the computation.
     */
    public NeighboursBruteForceFactory(Distance distance, int numThreads) {
        this.distance = distance;
        this.numThreads = numThreads;
        postConfig();
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static NeighboursBruteForceFactory deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        BruteForceFactoryProto queryProto = message.unpack(BruteForceFactoryProto.class);
        return new NeighboursBruteForceFactory(ProtoUtil.deserialize(queryProto.getDistance()),
                queryProto.getNumThreads());
    }

    @Override
    public NeighbourFactoryProto serialize() {
        return ProtoUtil.serialize(this);
    }

    /**
     * Constructs a brute-force nearest neighbor query object using the supplied array of {@link SGDVector}.
     * @param data An array of {@link SGDVector}.
     */
    @Override
    public NeighboursBruteForce createNeighboursQuery(SGDVector[] data) {
        return new NeighboursBruteForce(data, this.distance, this.numThreads);
    }

    @Override
    public Distance getDistance() {
        return distance;
    }

    @Override
    public int getNumThreads() {
        return numThreads;
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public synchronized void postConfig() {
        if (numThreads <= 0) {
            throw new PropertyException("numThreads", "The number of threads must be a number greater than 0.");
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        NeighboursBruteForceFactory that = (NeighboursBruteForceFactory) o;
        return numThreads == that.numThreads && distance.equals(that.distance);
    }

    @Override
    public int hashCode() {
        return Objects.hash(distance, numThreads);
    }
}
