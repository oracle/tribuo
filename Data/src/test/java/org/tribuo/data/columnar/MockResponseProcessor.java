/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.data.columnar;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.OutputFactory;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 *
 */
public class MockResponseProcessor implements ResponseProcessor<MockOutput> {

    @Config
    private String fieldName;

    private MockResponseProcessor() {}

    public MockResponseProcessor(String fieldName) {
        this.fieldName = fieldName;
    }

    @Override
    public OutputFactory<MockOutput> getOutputFactory() {
        return new MockOutputFactory();
    }

    @Deprecated
    @Override
    public String getFieldName() {
        return fieldName;
    }

    @Deprecated
    @Override
    public void setFieldName(String fieldName) {
        this.fieldName = fieldName;
    }

    @Deprecated
    @Override
    public Optional<MockOutput> process(String value) {
        return Optional.of(new MockOutput(value));
    }

    @Override
    public List<String> getFieldNames() {
        return Collections.singletonList(fieldName);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"ResponseProcessor");
    }
}
