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
import org.tribuo.math.distance.Distance;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.protos.NeighbourFactoryProto;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;

import java.io.Serializable;

/**
 * An interface for factories which create nearest neighbour query objects.
 */
public interface NeighboursQueryFactory extends Configurable, ProtoSerializable<NeighbourFactoryProto>, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Constructs a nearest neighbour query object using the supplied array of {@link SGDVector}.
     * @param data An array of {@link SGDVector}.
     * @return A query object.
     */
    NeighboursQuery createNeighboursQuery(SGDVector[] data);

    /**
     * Gets the {@link Distance} set on this object.
     * @return The distance function.
     */
    Distance getDistance();

    /**
     * Get the number of threads set on this object. There could be factory implementations that are sequential,
     * meaning they are single threaded.
     * @return The number of threads used to parallelize the query operation.
     */
    int getNumThreads();

    @Override
    default ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"NeighboursQueryFactory");
    }

    /**
     * Deserialization helper for NeighboursQueryFactories.
     * @param proto The proto to deserialize.
     * @return The query factory.
     */
    public static NeighboursQueryFactory deserialize(NeighbourFactoryProto proto) {
        return ProtoUtil.deserialize(proto);
    }
}
