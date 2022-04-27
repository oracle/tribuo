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

import java.util.Objects;

import org.tribuo.protos.core.MutableFeatureMapProto;
import org.tribuo.protos.core.VariableInfoProto;
import org.tribuo.util.ProtoUtil;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;

/**
 * A feature map that can record new feature value observations.
 */
@ProtoSerializableClass(serializedDataClass = MutableFeatureMapProto.class)
public class MutableFeatureMap extends FeatureMap {
    private static final long serialVersionUID = 2L;

    @ProtoSerializableField
    private final boolean convertHighCardinality;

    /**
     * Creates an empty feature map which converts high cardinality categorical variable infos into reals.
     * <p>
     * The conversion threshold is {@link CategoricalInfo#THRESHOLD}.
     */
    public MutableFeatureMap() {
        this(true);
    }

    /**
     * Creates an empty feature map which can optionally convert high cardinality categorical variable infos into reals.
     * <p>
     * The conversion threshold is {@link CategoricalInfo#THRESHOLD}.
     * @param convertHighCardinality Should this feature map convert high cardinality categorical variables into real variables?
     */
    public MutableFeatureMap(boolean convertHighCardinality) {
        super();
        this.convertHighCardinality = convertHighCardinality;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     */
    public static MutableFeatureMap deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        MutableFeatureMapProto proto = message.unpack(MutableFeatureMapProto.class);
        MutableFeatureMap obj = new MutableFeatureMap(proto.getConvertHighCardinality());
        for (VariableInfoProto infoProto : proto.getInfoList()) {
            VariableInfo info = ProtoUtil.deserialize(infoProto);
            Object o = obj.put(info);
            if (o != null) {
                throw new IllegalStateException("Invalid protobuf, found two mappings for " + info.getName());
            }
        }
        return obj;
    }

    /**
     * Adds a variable info into the feature map.
     * <p>
     * Returns the old one if there was a name collision, otherwise returns null.
     * @param info The info to add.
     * @return The old variable info or null.
     */
    public VariableInfo put(VariableInfo info) {
        VariableInfo old = m.put(info.getName(), info);
        return old;
    }

    /**
     * Adds an occurrence of a feature with a given name.
     *
     * @param name the name of the feature.
     * @param value the observed value of that feature.
     */
    public void add(String name, double value) {
        SkeletalVariableInfo info = (SkeletalVariableInfo) m.computeIfAbsent(name, CategoricalInfo::new);
        info.observe(value);

        // If there are too many categories, convert into a real info and drop the old categorical info.
        if (convertHighCardinality && info instanceof CategoricalInfo) {
            CategoricalInfo cInfo = (CategoricalInfo) info;
            if (cInfo.getUniqueObservations() > CategoricalInfo.THRESHOLD) {
                m.put(name,cInfo.generateRealInfo());
            }
        }
    }

    /**
     * Clears all the feature observations.
     */
    public void clear() {
        m.clear();
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
        MutableFeatureMap that = (MutableFeatureMap) o;
        return convertHighCardinality == that.convertHighCardinality;
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), convertHighCardinality);
    }

}
