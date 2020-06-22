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

package org.tribuo;

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
public interface Output<T extends Output<T>> extends Serializable {

    /**
     * Deep copy of the output up to it's immutable state.
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
}
