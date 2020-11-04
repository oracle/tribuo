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

package org.tribuo;

import com.oracle.labs.mlrg.olcut.provenance.Provenance;

import java.util.Collections;
import java.util.Map;

/**
 * Denotes this trainer emits a {@link SparseModel}.
 */
public interface SparseTrainer<T extends Output<T>> extends Trainer<T> {

    /**
     * Trains a sparse predictive model using the examples in the given data set.
     * @param examples The data set containing the examples.
     * @return A sparse predictive model that can be used to generate predictions for new examples.
     */
    @Override
    default public SparseModel<T> train(Dataset<T> examples) {
        return train(examples, Collections.emptyMap());
    }

    /**
     * Trains a sparse predictive model using the examples in the given data set.
     * @param examples the data set containing the examples.
     * @param runProvenance Training run specific provenance (e.g., fold number).
     * @return a predictive model that can be used to generate predictions for new examples.
     */
    @Override
    public SparseModel<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance);

}
