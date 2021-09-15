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

package org.tribuo.interop;

import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.SimpleDataSourceProvenance;

import java.time.OffsetDateTime;
import java.util.Map;

/**
 * A dummy provenance used to describe the dataset of external models.
 * <p>
 * Should not be used apart from by the external model system.
 */
public class ExternalDatasetProvenance extends DatasetProvenance {
    private static final long serialVersionUID = 1L;

    /**
     * An empty provenance used as a placeholder for externally trained models.
     * @param description The model description.
     * @param factory The output factory.
     * @param isSequence Is it a sequence model?
     * @param numFeatures The number of features.
     * @param numOutputs The output dimensionality.
     * @param <T> The type of the output.
     */
    public <T extends Output<T>> ExternalDatasetProvenance(String description, OutputFactory<T> factory, boolean isSequence, int numFeatures, int numOutputs) {
        super(new SimpleDataSourceProvenance(description, OffsetDateTime.now(),factory), new ListProvenance<>(), Dataset.class.getName(), false, isSequence, -1, numFeatures, numOutputs);
    }

    /**
     * Deserialization constructor.
     * @param map The provenances.
     */
    public ExternalDatasetProvenance(Map<String, Provenance> map) {
        super(map);
    }
}
