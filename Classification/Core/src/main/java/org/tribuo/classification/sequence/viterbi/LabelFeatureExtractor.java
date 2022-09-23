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

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.Feature;
import org.tribuo.classification.Label;
import org.tribuo.classification.protos.LabelFeatureExtractorProto;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;

import java.io.Serializable;
import java.util.List;

/**
 * A class for featurising labels from previous steps in Viterbi.
 */
public interface LabelFeatureExtractor extends Configurable, ProtoSerializable<LabelFeatureExtractorProto>, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Generates features based on the previously produced labels.
     * @param previousOutcomes The previous step's labels.
     * @param value The value to give to the features.
     * @return Features.
     */
    public List<Feature> extractFeatures(List<Label> previousOutcomes, double value);

    /**
     * Deserializes the LabelFeatureExtractor from the supplied protobuf.
     * @param proto The protobuf to deserialize.
     * @return The label feature extractor.
     */
    public static LabelFeatureExtractor deserialize(LabelFeatureExtractorProto proto) {
        return ProtoUtil.deserialize(proto);
    }
}

