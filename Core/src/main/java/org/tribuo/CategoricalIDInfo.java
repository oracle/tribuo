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

import com.oracle.labs.mlrg.olcut.util.MutableLong;

import java.util.Map;

/**
 * Same as a {@link CategoricalInfo}, but with an additional int id field.
 */
public class CategoricalIDInfo extends CategoricalInfo implements VariableIDInfo {
    private static final long serialVersionUID = 2L;

    private final int id;

    public CategoricalIDInfo(CategoricalInfo info, int id) {
        super(info);
        this.id = id;
    }

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
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double sum = 0.0;
        double sumSquares = 0.0;
        double mean;

        if (valueCounts != null) {
            for (Map.Entry<Double, MutableLong> e : valueCounts.entrySet()) {
                double value = e.getKey();
                double valCount = e.getValue().longValue();
                if (value > max) {
                    max = value;
                }
                if (value < min) {
                    min = value;
                }
                sum += value * valCount;
            }
            mean = sum / count;

            for (Map.Entry<Double, MutableLong> e : valueCounts.entrySet()) {
                double value = e.getKey();
                double valCount = e.getValue().longValue();
                sumSquares += (value - mean) * (value - mean) * valCount;
            }
        } else {
            min = observedValue;
            max = observedValue;
            mean = observedValue;
            sumSquares = 0.0;
        }

        return new RealIDInfo(name,count,max,min,mean,sumSquares,id);
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
