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

package org.tribuo.transform;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;

/**
 * An interface representing a class of transformations
 * which can be applied to a feature.
 * <p>
 * Transformations first have to be fitted, where they
 * gather the appropriate data statistics (e.g., min and max value).
 * The {@link TransformStatistics} can then be converted
 * into a {@link Transformer} which can apply the transform
 * to a feature value.
 * <p>
 * Transformations are configurable, but not serializable.
 * Describe them in config files or code.
 */
public interface Transformation extends Configurable, Provenancable<TransformationProvenance> {

    /**
     * Creates the statistics object for this Transformation.
     * @return The statistics object.
     */
    public TransformStatistics createStats();

}
