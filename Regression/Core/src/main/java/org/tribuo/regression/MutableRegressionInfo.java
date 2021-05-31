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

package org.tribuo.regression;

import com.oracle.labs.mlrg.olcut.util.MutableDouble;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.MutableOutputInfo;

import java.util.Map;

/**
 * A {@link MutableOutputInfo} for {@link Regressor}s. All observed Regressors must
 * contain the same named dimensions.
 */
public class MutableRegressionInfo extends RegressionInfo implements MutableOutputInfo<Regressor> {
    private static final long serialVersionUID = 2L;

    MutableRegressionInfo() {
        super();
    }

    /**
     * Constructs a mutable copy of the supplied regression info.
     * @param info The info to copy.
     */
    public MutableRegressionInfo(RegressionInfo info) {
        super(info);
    }

    @Override
    public void observe(Regressor output) {
        if (output == RegressionFactory.UNKNOWN_REGRESSOR) {
            unknownCount++;
        } else {
            if (overallCount != 0) {
                // Validate that the dimensions in this regressor are the same as the ones already observed.
                String[] names = output.getNames();
                if (names.length != countMap.size()) {
                    throw new IllegalArgumentException("Expected this Regressor to contain " + countMap.size() + " dimensions, found " + names.length);
                }
                for (String name : names) {
                    if (!countMap.containsKey(name)) {
                        throw new IllegalArgumentException("Regressor contains unexpected dimension named '" +name + "'");
                    }
                }
            }
            for (Regressor.DimensionTuple r : output) {
                String name = r.getName();
                double value = r.getValue();

                // Update max and min
                minMap.merge(name, new MutableDouble(value), (a, b) -> a.doubleValue() < b.doubleValue() ? a : b);
                maxMap.merge(name, new MutableDouble(value), (a, b) -> a.doubleValue() > b.doubleValue() ? a : b);

                // Update count
                MutableLong countValue = countMap.computeIfAbsent(name, k -> new MutableLong());
                countValue.increment();

                // Update mean
                MutableDouble meanValue = meanMap.computeIfAbsent(name, k -> new MutableDouble());
                double delta = value - meanValue.doubleValue();
                meanValue.increment(delta / countValue.longValue());

                // Update running sum of squares
                double delta2 = value - meanValue.doubleValue();
                MutableDouble sumSquaresValue = sumSquaresMap.computeIfAbsent(name, k -> new MutableDouble());
                sumSquaresValue.increment(delta * delta2);
            }
            overallCount++;
        }
    }

    @Override
    public void clear() {
        maxMap.clear();
        minMap.clear();
        meanMap.clear();
        sumSquaresMap.clear();
        countMap.clear();
    }

    @Override
    public MutableRegressionInfo copy() {
        return new MutableRegressionInfo(this);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("RegressionInfo(");
        for (Map.Entry<String,MutableLong> e : countMap.entrySet()) {
            String name = e.getKey();
            long count = e.getValue().longValue();
            builder.append(String.format("{name=%s,count=%d,max=%f,min=%f,mean=%f,variance=%f},",
                    name,
                    count,
                    maxMap.get(name).doubleValue(),
                    minMap.get(name).doubleValue(),
                    meanMap.get(name).doubleValue(),
                    (sumSquaresMap.get(name).doubleValue() / (count - 1))
            ));
        }
        builder.deleteCharAt(builder.length()-1);
        builder.append(")");
        return builder.toString();
    }

    @Override
    public String toReadableString() {
        return toString();
    }
}
