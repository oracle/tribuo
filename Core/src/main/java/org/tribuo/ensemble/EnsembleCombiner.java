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

package org.tribuo.ensemble;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;

import java.io.Serializable;
import java.util.List;

/**
 * An interface for combining predictions. Implementations should be final and immutable.
 */
public interface EnsembleCombiner<T extends Output<T>> extends Configurable, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Combine the predictions.
     * @param outputInfo The output domain.
     * @param predictions The predictions to combine.
     * @return The ensemble prediction.
     */
    public Prediction<T> combine(ImmutableOutputInfo<T> outputInfo, List<Prediction<T>> predictions);

    /**
     * Combine the supplied predictions. predictions.size() must equal weights.length.
     * @param outputInfo The output domain.
     * @param predictions The predictions to combine.
     * @param weights The weights to use for each prediction.
     * @return The ensemble prediction.
     */
    public Prediction<T> combine(ImmutableOutputInfo<T> outputInfo, List<Prediction<T>> predictions, float[] weights);

}
