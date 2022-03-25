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

/**
 * Same as a {@link CategoricalInfo}, but with an additional int id field.
 */
public class CategoricalIDInfo extends CategoricalInfo implements VariableIDInfo {
    private static final long serialVersionUID = 2L;

    private final int id;

    /**
     * Constructs a categorical id info copying the information from the supplied info, with the specified id.
     * @param info The info to copy.
     * @param id The id number to use.
     */
    public CategoricalIDInfo(CategoricalInfo info, int id) {
        super(info);
        this.id = id;
    }

    /**
     * Constructs a copy of the supplied categorical id info with the new name.
     * <p>
     * Used in the feature hashing system.
     * @param info The info to copy.
     * @param newName The new feature name.
     */
    private CategoricalIDInfo(CategoricalIDInfo info, String newName) {
        super(info,newName);
        this.id = info.id;
    }

    @Override
    public int getID() {
        return id;
    }

    /**
     * Generates a {@link RealIDInfo} that matches this CategoricalInfo and
     * also contains an id number.
     */
    @Override
    public RealIDInfo generateRealInfo() {
        RealInfo realInfo = super.generateRealInfo();
        return new RealIDInfo(realInfo,id);
    }

    @Override
    public CategoricalIDInfo copy() {
        return new CategoricalIDInfo(this,name);
    }

    @Override
    public CategoricalIDInfo makeIDInfo(int id) {
        return new CategoricalIDInfo(this,id);
    }

    @Override
    public CategoricalIDInfo rename(String newName) {
        return new CategoricalIDInfo(this,newName);
    }

    @Override
    public String toString() {
        if (valueCounts != null) {
            return "CategoricalFeature(name=" + name + ",id=" + id + ",count=" + count + ",map=" + valueCounts.toString() + ")";
        } else {
            return "CategoricalFeature(name=" + name + ",id=" + id + ",count=" + count + ",map={" +observedValue+","+observedCount+"})";
        }
    }
}
