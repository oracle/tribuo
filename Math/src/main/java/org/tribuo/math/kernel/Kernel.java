/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math.kernel;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.protos.KernelProto;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;

import java.io.Serializable;

/**
 * An interface for a Mercer kernel function.
 * <p>
 * It's preferable for kernels to override toString.
 */
public interface Kernel extends Configurable, ProtoSerializable<KernelProto>, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Calculates the similarity between two {@link SparseVector}s.
     * @param first The first SparseVector.
     * @param second The second SparseVector.
     * @return A value between 0 and 1, where 1 is most similar and 0 is least similar.
     */
    public double similarity(SparseVector first, SparseVector second);

    /**
     * Deserializes the kernel from the supplied protobuf.
     * @param proto The protobuf to deserialize.
     * @return The kernel.
     */
    public static Kernel deserialize(KernelProto proto) {
        return ProtoUtil.deserialize(proto);
    }
}
