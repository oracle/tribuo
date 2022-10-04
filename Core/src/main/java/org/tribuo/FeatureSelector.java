/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.provenance.FeatureSelectorProvenance;

/**
 * An interface for feature selection algorithms.
 * @param <T> The type of the output in the examples.
 */
public interface FeatureSelector<T extends Output<T>> extends Configurable, Provenancable<FeatureSelectorProvenance> {

    /**
     * Constant which denotes a full feature ranking should be generated rather than a subset.
     */
    public static final int SELECT_ALL = -1;

    /**
     * Does this feature selection algorithm return an ordered feature set?
     * @return True if the set is ordered.
     */
    public boolean isOrdered();

    /**
     * Selects features according to this selection algorithm from the specified dataset.
     * @param dataset The dataset to use.
     * @return A selected feature set.
     */
    public SelectedFeatureSet select(Dataset<T> dataset);

}
