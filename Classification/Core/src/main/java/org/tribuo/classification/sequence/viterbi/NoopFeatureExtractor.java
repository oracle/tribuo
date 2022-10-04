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

package org.tribuo.classification.sequence.viterbi;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Feature;
import org.tribuo.classification.Label;
import org.tribuo.classification.protos.LabelFeatureExtractorProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoUtil;

import java.util.Collections;
import java.util.List;

/**
 * A label feature extractor that doesn't produce any label based features.
 * <p>
 * It always returns {@link Collections#emptyList()}.
 */
@ProtoSerializableClass(version = NoopFeatureExtractor.CURRENT_VERSION)
public class NoopFeatureExtractor implements LabelFeatureExtractor {

    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Creates a new {@link LabelFeatureExtractor} that doesn't produce any label based features.
     */
    public NoopFeatureExtractor() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static NoopFeatureExtractor deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new NoopFeatureExtractor();
    }
    @Override
    public List<Feature> extractFeatures(List<Label> previousOutcomes, double value) {
        return Collections.emptyList();
    }

    @Override
    public String toString() {
        return "NoopFeatureExtractor";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "LabelFeatureExtractor");
    }

    @Override
    public LabelFeatureExtractorProto serialize() {
        return ProtoUtil.serialize(this);
    }
}

