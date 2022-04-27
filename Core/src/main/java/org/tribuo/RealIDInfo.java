/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;

import org.tribuo.protos.core.CategoricalInfoProto;
import org.tribuo.protos.core.RealIDInfoProto;
import org.tribuo.protos.core.RealInfoProto;
import org.tribuo.protos.core.VariableInfoProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;

import java.util.Objects;

/**
 * Same as a {@link RealInfo}, but with an additional int id field.
 */
@ProtoSerializableClass(serializedDataClass = RealIDInfoProto.class)
public class RealIDInfo extends RealInfo implements VariableIDInfo {
    private static final long serialVersionUID = 1L;

    @ProtoSerializableField
    private final int id;

    /**
     * Constructs a real id info from the supplied arguments.
     * @param name The feature name.
     * @param count The feature occurrence count.
     * @param max The maximum observed value.
     * @param min The minimum observed value.
     * @param mean The observed mean.
     * @param sumSquares The observed sum of squared values.
     * @param id The id number.
     */
    public RealIDInfo(String name, int count, double max, double min, double mean, double sumSquares, int id) {
        super(name,count,max,min,mean,sumSquares);
        this.id = id;
    }

    /**
     * Constructs a deep copy of the supplied real info and id.
     * @param info The info to copy.
     * @param id The new id number.
     */
    public RealIDInfo(RealInfo info, int id) {
        super(info);
        this.id = id;
    }

    /**
     * Copies the supplied real id info, renaming the feature.
     * @param info The info to copy.
     * @param newName The new name.
     */
    private RealIDInfo(RealIDInfo info, String newName) {
        super(info,newName);
        this.id = info.id;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     */
    public static RealIDInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        RealIDInfoProto proto = message.unpack(RealIDInfoProto.class);
        if (proto.getId() == -1) {
            throw new IllegalStateException("Invalid protobuf, found no id where one was expected.");
        }
        if (proto.getMax() < proto.getMin()) {
            throw new IllegalStateException("Invalid protobuf, min greater than max.");
        }
        if (proto.getMean() > proto.getMax()) {
            throw new IllegalStateException("Invalid protobuf, mean greater than max.");
        }
        if (proto.getMean() < proto.getMin()) {
            throw new IllegalStateException("Invalid protobuf, mean less than min.");
        }
        RealIDInfo info = new RealIDInfo(proto.getName(),proto.getCount(),
                proto.getMax(),proto.getMin(),
                proto.getMean(),proto.getSumSquares(),
                proto.getId());
        return info;
    }

    @Override
    public int getID() {
        return id;
    }

    @Override
    public RealIDInfo makeIDInfo(int id) {
        return new RealIDInfo(this,id);
    }

    @Override
    public RealIDInfo rename(String newName) {
        return new RealIDInfo(this,newName);
    }

    @Override
    public RealIDInfo copy() {
        return new RealIDInfo(this,name);
    }

    @Override
    public String toString() {
        return String.format("RealFeature(name=%s,id=%d,count=%d,max=%f,min=%f,mean=%f,variance=%f)",name,id,count,max,min,mean,(sumSquares /(count-1)));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        if (!super.equals(o)) {
            return false;
        }
        RealIDInfo that = (RealIDInfo) o;
        return id == that.id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), id);
    }

}
