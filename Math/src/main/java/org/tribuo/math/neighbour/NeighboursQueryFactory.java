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

package org.tribuo.math.neighbour;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.math.la.SGDVector;

/**
 * An interface for factories which create nearest neighbour query objects.
 */
public interface NeighboursQueryFactory extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Constructs a nearest neighbour query object using the supplied array of {@link SGDVector}.
     * @param data An array of {@link SGDVector}.
     */
    NeighboursQuery createNeighboursQuery(SGDVector[] data);

    @Override
    default ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"NeighboursQueryFactory");
    }
}
