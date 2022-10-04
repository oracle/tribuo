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

package org.tribuo.regression;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.MutableDouble;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.MutableOutputInfo;
import org.tribuo.protos.core.OutputDomainProto;
import org.tribuo.regression.protos.MutableRegressionInfoProto;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

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

    /**
     * Deserialization constructor.
     * @param countMap The dimension observation counts.
     * @param maxMap The max values per dimension.
     * @param minMap The min values per dimension.
     * @param meanMap The mean values per dimension.
     * @param sumSquaresMap The sum of squares per dimension.
     * @param unknownCount The number of unknowns observed.
     * @param overallCount The total number of things observed.
     */
    private MutableRegressionInfo(Map<String,MutableLong> countMap, Map<String,MutableDouble> maxMap, Map<String,MutableDouble> minMap, Map<String,MutableDouble> meanMap, Map<String,MutableDouble> sumSquaresMap, int unknownCount, long overallCount) {
        super(countMap,maxMap,minMap,meanMap,sumSquaresMap,unknownCount,overallCount);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static MutableRegressionInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        MutableRegressionInfoProto proto = message.unpack(MutableRegressionInfoProto.class);
        if ((proto.getLabelCount() != proto.getMaxCount()) || (proto.getLabelCount() != proto.getMinCount())
                || (proto.getLabelCount() != proto.getMeanCount()) || (proto.getLabelCount() != proto.getSumSquaresCount())
                || (proto.getLabelCount() != proto.getCountCount())) {
            throw new IllegalArgumentException("Invalid protobuf, expected the same number of dimension names, maxes," +
                    " mins, means, sumSquares and counts, found " + proto.getLabelCount() + " names, "
                    + proto.getMaxCount() + " maxes, " + proto.getMinCount() + " mins, " + proto.getMeanCount() + " means, "
                    + proto.getSumSquaresCount() + " sumSquares, and " + proto.getCountCount() + " counts.");
        }
        Map<String,MutableDouble> maxMap = new LinkedHashMap<>();
        Map<String,MutableDouble> minMap = new LinkedHashMap<>();
        Map<String,MutableDouble> meanMap = new LinkedHashMap<>();
        Map<String,MutableDouble> sumSquaresMap = new LinkedHashMap<>();
        Map<String,MutableLong> countMap = new TreeMap<>();
        for (int i = 0; i < proto.getLabelCount(); i++) {
            String lbl = proto.getLabel(i);
            long cnt = proto.getCount(i);
            MutableLong old = countMap.put(lbl,new MutableLong(cnt));
            if (old != null) {
                throw new IllegalArgumentException("Invalid protobuf, two mappings for " + lbl);
            }
            maxMap.put(lbl,new MutableDouble(proto.getMax(i)));
            minMap.put(lbl,new MutableDouble(proto.getMin(i)));
            meanMap.put(lbl,new MutableDouble(proto.getMean(i)));
            sumSquaresMap.put(lbl,new MutableDouble(proto.getSumSquares(i)));
        }
        return new MutableRegressionInfo(countMap,maxMap,minMap,meanMap,sumSquaresMap,proto.getUnknownCount(),proto.getOverallCount());
    }

    @Override
    public OutputDomainProto serialize() {
        OutputDomainProto.Builder outputBuilder = OutputDomainProto.newBuilder();

        outputBuilder.setClassName(MutableRegressionInfo.class.getName());
        outputBuilder.setVersion(0);

        MutableRegressionInfoProto.Builder data = MutableRegressionInfoProto.newBuilder();
        for (Map.Entry<String, MutableLong> e : countMap.entrySet()) {
            data.addLabel(e.getKey());
            data.addCount(e.getValue().longValue());
            data.addMax(maxMap.get(e.getKey()).doubleValue());
            data.addMin(minMap.get(e.getKey()).doubleValue());
            data.addMean(meanMap.get(e.getKey()).doubleValue());
            data.addSumSquares(sumSquaresMap.get(e.getKey()).doubleValue());
        }
        data.setUnknownCount(unknownCount);
        data.setOverallCount(overallCount);

        outputBuilder.setSerializedData(Any.pack(data.build()));

        return outputBuilder.build();
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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MutableRegressionInfo that = (MutableRegressionInfo) o;
        if (unknownCount == that.unknownCount && overallCount == that.overallCount) {
            for (Map.Entry<String,MutableLong> e : countMap.entrySet()) {
                MutableLong other = that.countMap.get(e.getKey());
                if (other == null || (other.longValue() != e.getValue().longValue())) {
                    return false;
                } else {
                    // mapping exists, check max, min, mean, sumSquares
                    if (!checkMutableDouble(maxMap.get(e.getKey()), that.maxMap.get(e.getKey()))) {
                        return false;
                    }
                    if (!checkMutableDouble(minMap.get(e.getKey()), that.minMap.get(e.getKey()))) {
                        return false;
                    }
                    if (!checkMutableDouble(meanMap.get(e.getKey()), that.meanMap.get(e.getKey()))) {
                        return false;
                    }
                    if (!checkMutableDouble(sumSquaresMap.get(e.getKey()), that.sumSquaresMap.get(e.getKey()))) {
                        return false;
                    }
                }
            }
            return true;
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        return Objects.hash(countMap, maxMap, minMap, meanMap, sumSquaresMap, unknownCount, overallCount);
    }
}
