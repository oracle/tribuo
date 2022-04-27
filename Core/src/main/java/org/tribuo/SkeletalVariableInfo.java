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

import java.util.Objects;
import java.util.logging.Logger;

import org.tribuo.protos.ProtoSerializableField;

/**
 * Contains information about a feature and can be stored in the feature map
 * in a {@link Dataset}.
 */
public abstract class SkeletalVariableInfo implements VariableInfo {
    private static final long serialVersionUID = 2L;

    private static final Logger logger = Logger.getLogger(SkeletalVariableInfo.class.getName());

    /**
     * The name of the feature.
     */
    @ProtoSerializableField
    protected final String name;
    
    /**
     * How often the feature occurs in the dataset.
     */
    @ProtoSerializableField
    protected int count;

    /**
     * Constructs a variable info with the supplied name.
     * @param name The feature name.
     */
    protected SkeletalVariableInfo(String name) {
        this.name = name;
    }

    /**
     * Constructs a variable info with the supplied name and initial count.
     * @param name The feature name.
     * @param count The initial occurrence count.
     */
    protected SkeletalVariableInfo(String name, int count) {
        this.name = name;
        this.count = count;
    }

    /**
     * Records the value.
     * @param value The observed value.
     */
    protected void observe(double value) {
        count++;
    }

    /**
     * Returns the name of the feature.
     * @return The name of the feature.
     */
    @Override
    public String getName() {
        return name;
    }

    /**
     * Returns the occurrence count of this feature.
     * @return The count of observed values.
     */
    @Override
    public int getCount() {
        return count;
    }

    @Override
    public String toString() {
        return "Feature(name="+name+",count="+count+")";
    }

    @Override
    public int hashCode() {
        int hash = 3;
        hash = 37 * hash + this.name.hashCode();
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final SkeletalVariableInfo other = (SkeletalVariableInfo) obj;
        return Objects.equals(this.name, other.name);
    }
}
