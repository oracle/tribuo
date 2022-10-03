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

package org.tribuo.test;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.Output;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.OutputProto;
import org.tribuo.test.protos.MockOutputProto;

import java.util.Objects;

/**
 * An Output for use in tests which is very similar to Label.
 */
@ProtoSerializableClass(serializedDataClass=MockOutputProto.class, version=0)
public class MockOutput implements Output<MockOutput> {
    private static final long serialVersionUID = 1L;

    @ProtoSerializableField
    public final String label;

    public MockOutput(String label) {
        this.label = label;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static MockOutput deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        MockOutputProto proto = message.unpack(MockOutputProto.class);
        MockOutput info = new MockOutput(proto.getLabel());
        return info;
    }

    @Override
    public OutputProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public MockOutput copy() {
        return new MockOutput(label);
    }

    @Override
    public String getSerializableForm(boolean includeConfidence) {
        return label;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof MockOutput)) return false;

        MockOutput that = (MockOutput) o;

        return label != null ? label.equals(that.label) : that.label == null;
    }

    @Override
    public boolean fullEquals(MockOutput other) {
        return label.equals(other.label);
    }

    @Override
    public String toString() {
        return label;
    }

    @Override
    public int hashCode() {
        return Objects.hash(label);
    }
}

