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

package org.tribuo.provenance.impl;

import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.provenance.DatasetProvenance;

import java.util.Map;

/**
 * An empty DatasetProvenance, should not be used except by the provenance removal system.
 */
public final class EmptyDatasetProvenance extends DatasetProvenance {
    private static final long serialVersionUID = 1L;

    /**
     * An empty dataset provenance.
     */
    public EmptyDatasetProvenance() {
        super(new EmptyDataSourceProvenance(), new ListProvenance<>(), Dataset.class.getName(), false, false, -1, -1, -1);
    }

    /**
     * Deserialization constructor.
     * @param map The provenance map, which is ignored as this provenance is empty.
     */
    public EmptyDatasetProvenance(Map<String, Provenance> map) {
        this();
    }

    @Override
    public String toString() {
        return generateString("EmptyDataset");
    }
}
