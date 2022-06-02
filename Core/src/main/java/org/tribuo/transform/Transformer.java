/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

import org.tribuo.protos.core.TransformerProto;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;

import java.io.Serializable;

/**
 * A fitted {@link Transformation} which can apply
 * a transform to the input value. Usually contains
 * feature specific information from the training data.
 * <p>
 * Transformers are serializable, and should only
 * be constructed by their {@link TransformStatistics}.
 */
public interface Transformer extends ProtoSerializable<TransformerProto>, Serializable {

    /**
     * Applies the transformation to the supplied
     * input value.
     * @param input The value to transform.
     * @return The transformed value.
     */
    public double transform(double input);

    /**
     * Deserializes a {@link TransformerProto} into a {@link Transformer} subclass.
     * @param proto The proto to deserialize.
     * @return The deserialized FeatureMap.
     */
    public static Transformer deserialize(TransformerProto proto) {
        return ProtoUtil.deserialize(proto);
    }
}
