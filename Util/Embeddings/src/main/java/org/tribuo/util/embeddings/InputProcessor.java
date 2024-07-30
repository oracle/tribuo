/*
 * Copyright (c) 2023, 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.util.embeddings;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensorLike;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 *
 */
public interface InputProcessor extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    boolean validate(Map<String, NodeInfo> inputInfo);

    Set<String> getVocab();

    long getTokenId(String token);

    int getMaxLength();

    ProcessedInput process(OrtEnvironment env, List<String> input)
        throws OrtException;

    default ProcessedInput process(OrtEnvironment env, String input) throws OrtException {
        return process(env, List.of(input));
    }

    /**
     * A record containing the inputs ready for ORT along with the example lengths before padding.
     * @param inputs The ORT inputs.
     * @param tokenLengths The example lengths before padding.
     */
    public record ProcessedInput(Map<String, ? extends OnnxTensorLike> inputs,
                                 long[] tokenLengths,
                                 LongTensor tokenIds) {}

}
