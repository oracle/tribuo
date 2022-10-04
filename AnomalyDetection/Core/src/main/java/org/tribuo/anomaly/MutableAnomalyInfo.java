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

package org.tribuo.anomaly;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.MutableOutputInfo;
import org.tribuo.anomaly.protos.AnomalyInfoProto;
import org.tribuo.protos.ProtoSerializableClass;

/**
 * An {@link MutableOutputInfo} object for {@link Event}s.
 * <p>
 * Counts the number of {@link Event.EventType#ANOMALOUS}, {@link Event.EventType#EXPECTED}
 * and {@link Event.EventType#UNKNOWN} outputs observed. The unknown output is invalid
 * at training time, and used as a prediction time sentinel (similarly to other Tribuo
 * prediction tasks).
 * <p>
 * Anomaly detection has a fixed domain, so it will throw {@link IllegalArgumentException}
 * if you somehow modify the {@link Event.EventType} enum to add a new value.
 */
@ProtoSerializableClass(serializedDataClass=AnomalyInfoProto.class, version=0)
public final class MutableAnomalyInfo extends AnomalyInfo implements MutableOutputInfo<Event> {
    private static final long serialVersionUID = 1L;

    MutableAnomalyInfo() {
        super();
    }

    MutableAnomalyInfo(AnomalyInfo info) {
        super(info);
    }

    private MutableAnomalyInfo(long expectedCount, long anomalyCount, int unknownCount) {
        super(expectedCount,anomalyCount,unknownCount);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static MutableAnomalyInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        AnomalyInfoProto proto = message.unpack(AnomalyInfoProto.class);
        return new MutableAnomalyInfo(proto.getExpectedCount(),proto.getAnomalyCount(),proto.getUnknownCount());
    }

    @Override
    public void observe(Event output) {
        if (output == AnomalyFactory.UNKNOWN_EVENT) {
            unknownCount++;
        } else {
            switch (output.getType()) {
                case ANOMALOUS:
                    anomalyCount++;
                    break;
                case EXPECTED:
                    expectedCount++;
                    break;
                default:
                    throw new IllegalArgumentException("Unexpected EventType, found " + output.getType());
            }
        }
    }

    @Override
    public void clear() {
        unknownCount = 0;
        anomalyCount = 0;
        expectedCount = 0;
    }

    @Override
    public MutableAnomalyInfo copy() {
        return new MutableAnomalyInfo(this);
    }
}
