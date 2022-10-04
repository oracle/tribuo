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
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.core.CategoricalIDInfoProto;

import java.util.HashMap;
import java.util.List;
import java.util.Objects;

/**
 * Same as a {@link CategoricalInfo}, but with an additional int id field.
 */
@ProtoSerializableClass(version = CategoricalIDInfo.CURRENT_VERSION, serializedDataClass = CategoricalIDInfoProto.class)
public class CategoricalIDInfo extends CategoricalInfo implements VariableIDInfo {
    private static final long serialVersionUID = 2L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    @ProtoSerializableField
    private final int id;

    /**
     * Constructs a categorical id info copying the information from the supplied info, with the specified id.
     * @param info The info to copy.
     * @param id The id number to use.
     */
    public CategoricalIDInfo(CategoricalInfo info, int id) {
        super(info);
        if (id < 0) {
            throw new IllegalArgumentException("Invalid id number, must be non-negative, found " + id);
        }
        this.id = id;
    }

    /**
     * Constructs a copy of the supplied categorical id info with the new name.
     * <p>
     * Used in the feature hashing system.
     * @param info The info to copy.
     * @param newName The new feature name.
     */
    private CategoricalIDInfo(CategoricalIDInfo info, String newName) {
        super(info,newName);
        this.id = info.id;
    }

    /**
     * Deserialization constructor.
     * @param name The info name.
     * @param id The info id.
     */
    private CategoricalIDInfo(String name, int id) {
        super(name);
        if (id < 0) {
            throw new IllegalArgumentException("Invalid id number, must be non-negative, found " + id);
        }
        this.id = id;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static CategoricalIDInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        CategoricalIDInfoProto proto = message.unpack(CategoricalIDInfoProto.class);
        CategoricalIDInfo info = new CategoricalIDInfo(proto.getName(),proto.getId());
        List<Double> keys = proto.getKeyList();
        List<Long> values = proto.getValueList();
        if (keys.size() != values.size()) {
            throw new IllegalStateException("Invalid protobuf, keys and values don't match. keys.size() = " + keys.size() + ", values.size() = " + values.size());
        }
        int newCount = 0;
        if (keys.size() > 1) {
            info.valueCounts = new HashMap<>(keys.size());
            for (int i = 0; i < keys.size(); i++) {
                if (values.get(i) < 0) {
                    throw new IllegalStateException("Invalid protobuf, counts must be positive, found " + values.get(i) + " for value " + keys.get(i));
                }
                info.valueCounts.put(keys.get(i),new MutableLong(values.get(i)));
                newCount += values.get(i).intValue();
            }
        } else {
            info.observedValue = proto.getObservedValue();
            info.observedCount = proto.getObservedCount();
            newCount = (int) proto.getObservedCount();
            if (info.observedCount < 0) {
                throw new IllegalStateException("Invalid protobuf, counts must be positive, found " + info.observedCount + " for value " + info.observedValue);
            }
        }
        if (newCount != proto.getCount()) {
            throw new IllegalStateException("Invalid protobuf, count " + newCount + " did not match expected value " + proto.getCount());
        }
        info.count = newCount;
        return info;
    }

    @Override
    public int getID() {
        return id;
    }

    /**
     * Generates a {@link RealIDInfo} that matches this CategoricalInfo and
     * also contains an id number.
     */
    @Override
    public RealIDInfo generateRealInfo() {
        RealInfo realInfo = super.generateRealInfo();
        return new RealIDInfo(realInfo,id);
    }

    @Override
    public CategoricalIDInfo copy() {
        return new CategoricalIDInfo(this,name);
    }

    @Override
    public CategoricalIDInfo makeIDInfo(int id) {
        return new CategoricalIDInfo(this,id);
    }

    @Override
    public CategoricalIDInfo rename(String newName) {
        return new CategoricalIDInfo(this,newName);
    }

    @Override
    public String toString() {
        if (valueCounts != null) {
            return "CategoricalFeature(name=" + name + ",id=" + id + ",count=" + count + ",map=" + valueCounts.toString() + ")";
        } else {
            return "CategoricalFeature(name=" + name + ",id=" + id + ",count=" + count + ",map={" +observedValue+","+observedCount+"})";
        }
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
        CategoricalIDInfo that = (CategoricalIDInfo) o;
        return id == that.id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), id);
    }

}
