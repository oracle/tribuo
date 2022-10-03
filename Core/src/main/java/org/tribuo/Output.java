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

package org.tribuo;

import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.OutputProto;

import java.io.Serializable;

/**
 * Output is the root interface for the supported prediction types.
 * <p>
 * It's subclassed in each of the modules:
 * <ul>
 * <li>Label for multi-class classification</li>
 * <li>MultiLabel for multi-label classification</li>
 * <li>ClusterID for clustering</li>
 * <li>Regressor for regression</li>
 * <li>Event for anomaly detection</li>
 * </ul>
 * Equals and hashcode are defined to only look at the strings stored in an Output, not any score
 * values. For equality that takes into account the scores, use {@link Output#fullEquals}.
 */
public interface Output<T extends Output<T>> extends ProtoSerializable<OutputProto>, Serializable {

    /**
     * Deep copy of the output up to its immutable state.
     * @return A copy of the output.
     */
    public T copy();

    /**
     * Generates a String suitable for writing to a csv or json file.
     * @param includeConfidence Include whatever confidence score the label contains, if known.
     * @return A String representation of this Output.
     */
    public String getSerializableForm(boolean includeConfidence);

    /**
     * Compares other to this output. Uses all score values
     * and the strings.
     * @param other Another output instance.
     * @return True if the other instance has value equality to this instance. False otherwise.
     */
    public boolean fullEquals(T other);

    /**
     * Compares other to this output. Uses all score values
     * and the strings.
     * <p>
     * The default implementation of this method ignores the tolerance for compatibility reasons,
     * it is overridden in all output classes in Tribuo.
     * @param other Another output instance.
     * @param tolerance The tolerance level for an absolute value comparison.
     * @return True if the other instance has value equality to this instance. False otherwise.
     */
    default public boolean fullEquals(T other, double tolerance) {
        return fullEquals(other);
    }

    /**
     * Deserializes a {@link OutputProto} into a {@link Output} subclass.
     * @param proto The proto to deserialize.
     * @return The deserialized Output.
     */
    public static Output<?> deserialize(OutputProto proto) {
        return ProtoUtil.deserialize(proto);
    }
}
