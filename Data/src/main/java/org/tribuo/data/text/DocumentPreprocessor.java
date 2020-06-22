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

package org.tribuo.data.text;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;

/**
 * An interface for things that can pre-process documents before they are 
 * broken into features.
 */
public interface DocumentPreprocessor extends Configurable, Provenancable<ConfiguredObjectProvenance> {
    /**
     * Processes the content of part of a document stored as a string, returning a 
     * new string. 
     * @param doc the document to process
     * @return the processed string. Note that the return value may be {@code null},
     * in which case the resulting string will be ignored.
     */
    public String processDoc(String doc);
}
