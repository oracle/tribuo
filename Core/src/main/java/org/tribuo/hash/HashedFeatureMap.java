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

package org.tribuo.hash;

import java.util.Map;
import java.util.TreeMap;

import org.tribuo.FeatureMap;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.ProtoSerializableClass;
import org.tribuo.ProtoSerializableField;
import org.tribuo.VariableIDInfo;
import org.tribuo.VariableInfo;
import org.tribuo.protos.core.FeatureDomainProto;
import org.tribuo.protos.core.HashedFeatureMapProto;
import org.tribuo.protos.core.HasherProto;
import org.tribuo.protos.core.ImmutableFeatureMapProto;
import org.tribuo.protos.core.VariableInfoProto;
import org.tribuo.util.ProtoUtil;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;

/**
 * A {@link FeatureMap} used by the {@link HashingTrainer} to
 * provide feature name hashing and guarantee that the {@link Model}
 * does not contain feature name information, but still works
 * with unhashed features names.
 */
@ProtoSerializableClass(serializedDataClass = HashedFeatureMapProto.class)
public final class HashedFeatureMap extends ImmutableFeatureMap {
    private static final long serialVersionUID = 1L;

    @ProtoSerializableField
    private final Hasher hasher;

    private HashedFeatureMap(Hasher hasher) {
        super();
        this.hasher = hasher;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     */
    public static HashedFeatureMap deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        HashedFeatureMapProto proto = message.unpack(HashedFeatureMapProto.class);
        HasherProto hasherProto = proto.getHasher();
        Hasher hasher = (Hasher) ProtoUtil.instantiate(hasherProto.getVersion(), hasherProto.getClassName(), hasherProto.getSerializedData());
        HashedFeatureMap obj = new HashedFeatureMap(hasher);
        for (VariableInfoProto infoProto : proto.getInfoList()) {
            VariableIDInfo info = (VariableIDInfo) ProtoUtil.instantiate(infoProto.getVersion(), infoProto.getClassName(), infoProto.getSerializedData());
            Object o = obj.idMap.put(info.getID(), info);
            Object otherO = obj.m.put(info.getName(),info);
            if ((o != null) || (otherO != null)) {
                throw new IllegalStateException("Invalid protobuf, found two mappings for " + info.getName());
            }
        }
        obj.size = proto.getInfoCount();
        return obj;
    }

    @Override
    public VariableIDInfo get(String name) {
        String hash = hasher.hash(name);
        return (VariableIDInfo) m.get(hash);
    }

    /**
     * Gets the id number for this feature, returns -1 if it's unknown.
     * @param name The name of the feature.
     * @return A non-negative integer if the feature is known, -1 otherwise.
     */
    @Override
    public int getID(String name) {
        VariableIDInfo info = get(name);
        if (info != null) {
            return info.getID();
        } else {
            return -1;
        }
    }

    /**
     * The salt is not serialised with the {@link Model}.
     * It must be set after deserialisation to the same value from training time.
     * <p>
     * If the salt is invalid it will throw {@link IllegalArgumentException}.
     * @param salt The salt value. Must be the same as the one from training time.
     */
    public void setSalt(String salt) {
        hasher.setSalt(salt);
    }

    /**
     * Converts a standard {@link FeatureMap} by hashing each entry
     * using the supplied hash function {@link Hasher}.
     * <p>
     * This preserves the index ordering of the original feature names,
     * which is important for making sure test time performance is good.
     * <p>
     * It guarantees any collisions will produce a feature id number lower
     * than the previous feature's number, and so can be easily removed.
     *
     * @param map The {@link FeatureMap} to hash.
     * @param hasher The hashing function.
     * @return A {@link HashedFeatureMap}.
     */
    public static HashedFeatureMap generateHashedFeatureMap(FeatureMap map, Hasher hasher) {
        HashedFeatureMap hashedMap = new HashedFeatureMap(hasher);
        TreeMap<String,VariableInfo> treeHashMap = new TreeMap<>();
        for (VariableInfo f : map) {
            String hash = hasher.hash(f.getName());
            if (!treeHashMap.containsKey(f.getName())) {
                VariableInfo newF = f.rename(hash);
                treeHashMap.put(f.getName(),newF);
            }
        }
        int counter = 0;
        for (Map.Entry<String,VariableInfo> e : treeHashMap.entrySet()) {
            VariableIDInfo newF = e.getValue().makeIDInfo(counter);
            if (!hashedMap.m.containsKey(newF.getName())) {
                hashedMap.m.put(newF.getName(), newF);
                hashedMap.idMap.put(newF.getID(), newF);
                counter++;
            }
        }
        hashedMap.size = hashedMap.m.size();
        return hashedMap;
    }

}
