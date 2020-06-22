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

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;

import java.util.List;

/**
 * Takes a list of columnar features and adds new features or removes existing features.
 * <p>
 * Any existing features have their names preserved if they are not removed.
 * <p>
 * New features that are created must use the {@link ColumnarFeature#CONJUNCTION} String
 * as their field name, as enforced by the two field {@link ColumnarFeature} constructor.
 */
public interface FeatureProcessor extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Processes a list of {@link ColumnarFeature}s, transforming it
     * by adding conjunctions or removing unnecessary features.
     * @param features The list of features to process.
     * @return A (possibly empty) list of features.
     */
    public List<ColumnarFeature> process(List<ColumnarFeature> features);

}
