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

package org.tribuo;

import org.tribuo.protos.core.VariableInfoProto;

import java.io.Serializable;
import java.util.SplittableRandom;

/**
 * A VariableInfo subclass contains information about a feature and
 * its observed values.
 */
public interface VariableInfo extends Serializable, ProtoSerializable<VariableInfoProto>, Cloneable {
    /**
     * The name of this feature.
     * @return The feature name.
     */
    public String getName();

    /**
     * The occurrence count of this feature.
     * @return The occurrence count.
     */
    public int getCount();

    /**
     * Generates a VariableIDInfo subclass which represents the same feature.
     * @param id The id number.
     * @return A VariableInfo with the same information, plus the id.
     */
    public VariableIDInfo makeIDInfo(int id);

    /**
     * Rename generates a fresh VariableInfo with the new name.
     *
     * The name forms part of the hashcode so it's immutable in the object.
     * @param name The new name.
     * @return A VariableInfo subclass with the new name.
     */
    public VariableInfo rename(String name);

    /**
     * Sample a value uniformly from the range of this variable.
     *
     * @param rng The rng to use.
     * @return A sample from this variable.
     */
    public double uniformSample(SplittableRandom rng);

    /**
     * Returns a copy of this variable info.
     * @return A copy.
     */
    public VariableInfo copy();
}
