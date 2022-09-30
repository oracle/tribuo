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

package org.tribuo.math.distance;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.protos.DistanceProto;
import org.tribuo.protos.ProtoSerializable;

import java.io.Serializable;

/**
 * Interface for distance functions.
 * <p>
 * Must be valid distance functions which are positive, symmetric, and obey the triangle inequality.
 */
public interface Distance extends Configurable, ProtoSerializable<DistanceProto>, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Computes the distance between the two vectors.
     * @param first The first vector.
     * @param second The second vector.
     * @return The distance between them.
     */
    public double computeDistance(SGDVector first, SGDVector second);

}
