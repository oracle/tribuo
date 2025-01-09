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

package org.tribuo.util.embeddings.processors;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.ValueInfo;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.util.embeddings.InputProcessor;
import org.tribuo.util.embeddings.LongTensorBuffer;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A no-op input processor which works with models that build in the tokenization and averaging.
 * <p>
 * This input processor returns an empty set for the vocab and max length as it handled by the model.
 */
public final class NoOpInputProcessor implements InputProcessor {
    private static final Logger logger = Logger.getLogger(NoOpInputProcessor.class.getName());

    public static final String DEFAULT_INPUT_NAME = "input";

    /**
     * String input name.
     */
    @Config(description="String input name")
    private String inputName = DEFAULT_INPUT_NAME;

    /**
     * Constructs a no-op input processor with the default name.
     */
    public NoOpInputProcessor() {
        this(DEFAULT_INPUT_NAME);
    }

    /**
     * Constructs a no-op input processor with the specified input name.
     * @param inputName The input name.
     */
    public NoOpInputProcessor(String inputName) {
        this.inputName = inputName;
    }

    @Override
    public boolean validate(Map<String, NodeInfo> inputInfo) {
        if (inputInfo.size() != 1 || !inputInfo.containsKey(inputName)) {
            logger.warning("Invalid model for this input processor, expected one input called '" + inputName + "', found " + inputInfo);
            return false;
        }
        ValueInfo info = inputInfo.get(inputName).getInfo();
        if (info instanceof TensorInfo ti) {
            if (ti.type != OnnxJavaType.STRING) {
                logger.warning("Invalid model for this input processor, expected a String input, found " + ti);
                return false;
            } else if (!Arrays.equals(new long[]{-1}, ti.getShape())) {
                logger.warning("Invalid model for this input processor, expected a single dimensional string tensor of unbound shape, found shape " + Arrays.toString(ti.getShape()));
                return false;
            }
            return true;
        } else {
            logger.warning("Invalid model for this input processor, expects a non-tensor input " + info);
            return false;
        }
    }

    @Override
    public Set<String> getVocab() {
        return Collections.emptySet();
    }

    @Override
    public long getTokenId(String token) {
        return 0;
    }

    @Override
    public int getMaxLength() {
        return 0;
    }

    @Override
    public ProcessedInput process(OrtEnvironment env, List<String> input) throws OrtException {
        String[] strArray = input.toArray(new String[0]);
        long[] tokenLengths = new long[strArray.length];

        return new ProcessedInput(Map.of(inputName,OnnxTensor.createTensor(env, strArray)), tokenLengths, new LongTensorBuffer(new long[]{1,1}));
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "InputProcessor");
    }
}
