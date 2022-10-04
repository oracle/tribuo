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
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.protos.core.OutputDomainProto;
import org.tribuo.regression.protos.ImmutableRegressionInfoProto;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A {@link ImmutableOutputInfo} for {@link Regressor}s.
 */
public class ImmutableRegressionInfo extends RegressionInfo implements ImmutableOutputInfo<Regressor> {
    private static final Logger logger = Logger.getLogger(ImmutableRegressionInfo.class.getName());

    private static final long serialVersionUID = 2L;

    private final Map<Integer,String> idLabelMap;

    private final Map<String,Integer> labelIDMap;

    private final Set<Regressor> domain;

    private transient double[] minArray;
    private transient double[] maxArray;
    private transient double[] meanArray;
    private transient double[] varianceArray;

    /**
     * Copies an ImmutableRegressionInfo including the ids.
     * @param info The info to copy.
     */
    private ImmutableRegressionInfo(ImmutableRegressionInfo info) {
        super(info);
        idLabelMap = new LinkedHashMap<>();
        idLabelMap.putAll(info.idLabelMap);
        labelIDMap = new LinkedHashMap<>();
        labelIDMap.putAll(info.labelIDMap);
        domain = calculateDomain(minMap);
        computeStatisticArrays();
    }

    /**
     * Builds an ImmutableRegressionInfo from the supplied RegressionInfo.
     * <p>
     * Enforces that the ids are generated in lexicographic order.
     * @param info The info to build from.
     */
    ImmutableRegressionInfo(RegressionInfo info) {
        super(info);
        idLabelMap = new LinkedHashMap<>();
        labelIDMap = new LinkedHashMap<>();
        // Sort count map
        SortedSet<String> names = new TreeSet<>(countMap.keySet());
        int counter = 0;
        for (String e : names) {
            idLabelMap.put(counter,e);
            labelIDMap.put(e,counter);
            counter++;
        }
        domain = calculateDomain(minMap);
        computeStatisticArrays();
    }

    /**
     * Builds an ImmutableRegressionInfo from the supplied RegressionInfo using the
     * supplied id mapping.
     * <p>
     * This can result in id mappings which are not in lexicographic order.
     * <p>
     * Expects that the mapping is made up of single dimension regressors, usually
     * {@link org.tribuo.regression.Regressor.DimensionTuple}, though it only enforces
     * that a single dimension is present.
     * @param info The info to build from.
     * @param mapping The id number - dimension name bijective mapping.
     */
    ImmutableRegressionInfo(RegressionInfo info, Map<Regressor,Integer> mapping) {
        super(info);
        if (mapping.size() != info.size()) {
            throw new IllegalStateException("Mapping and info come from different sources, mapping.size() = " + mapping.size() + ", info.size() = " + info.size());
        }

        String[] names = new String[mapping.size()];
        for (Map.Entry<Regressor, Integer> e : mapping.entrySet()) {
            Regressor r = e.getKey();
            String[] curNames = r.getNames();
            if (names[e.getValue()] != null) {
                throw new IllegalArgumentException("Mapping must be a bijection, but found two mappings for index " + e.getValue() + ", '" + names[e.getValue()] + "' and '" + curNames[0] + "'");
            }
            else if (curNames.length == 1) {
                names[e.getValue()] = curNames[0];
            } else {
                throw new IllegalArgumentException("Mapping must contain a single regression dimension per id, but contains " + Arrays.toString(names) + " -> " + e.getValue());
            }
        }
        idLabelMap = new LinkedHashMap<>();
        labelIDMap = new LinkedHashMap<>();
        for (int i = 0; i < names.length; i++) {
            idLabelMap.put(i,names[i]);
            labelIDMap.put(names[i],i);
        }
        if (!countMap.keySet().containsAll(labelIDMap.keySet()) || !labelIDMap.keySet().containsAll(countMap.keySet())) {
            throw new IllegalArgumentException("Mapping must contain an entry for each element in the info, found " + labelIDMap.keySet() + " and " + countMap.keySet());
        }
        domain = calculateDomain(minMap);
        computeStatisticArrays();
    }

    /**
     * Deserialization constructor.
     * @param countMap The dimension observation counts.
     * @param labelIDMap The dimension indices.
     * @param maxMap The max values per dimension.
     * @param minMap The min values per dimension.
     * @param meanMap The mean values per dimension.
     * @param sumSquaresMap The sum of squares per dimension.
     * @param unknownCount The number of unknowns observed.
     * @param overallCount The total number of things observed.
     */
    private ImmutableRegressionInfo(Map<String,MutableLong> countMap, Map<String,Integer> labelIDMap, Map<String,MutableDouble> maxMap, Map<String,MutableDouble> minMap, Map<String,MutableDouble> meanMap, Map<String,MutableDouble> sumSquaresMap, int unknownCount, long overallCount) {
        super(countMap,maxMap,minMap,meanMap,sumSquaresMap,unknownCount,overallCount);
        this.labelIDMap = new LinkedHashMap<>(labelIDMap);
        this.idLabelMap = new LinkedHashMap<>();
        for (Map.Entry<String,Integer> e : this.labelIDMap.entrySet()) {
            idLabelMap.put(e.getValue(),e.getKey());
        }
        this.domain = calculateDomain(this.minMap);
        computeStatisticArrays();
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static ImmutableRegressionInfo deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        ImmutableRegressionInfoProto proto = message.unpack(ImmutableRegressionInfoProto.class);
        if ((proto.getLabelCount() != proto.getMaxCount()) || (proto.getLabelCount() != proto.getMinCount())
                || (proto.getLabelCount() != proto.getMeanCount()) || (proto.getLabelCount() != proto.getSumSquaresCount())
                || (proto.getLabelCount() != proto.getCountCount()) || (proto.getLabelCount() != proto.getIdCount())) {
            throw new IllegalArgumentException("Invalid protobuf, expected the same number of dimension names, maxes," +
                    " mins, means, sumSquares, counts and ids found " + proto.getLabelCount() + " names, "
                    + proto.getMaxCount() + " maxes, " + proto.getMinCount() + " mins, " + proto.getMeanCount() + " means, "
                    + proto.getSumSquaresCount() + " sumSquares, " + proto.getCountCount() + " counts, and "
                    + proto.getIdCount() + "ids.");
        }
        Map<String,Integer> labelIDMap = new LinkedHashMap<>();
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
            labelIDMap.put(lbl,proto.getId(i));
            maxMap.put(lbl,new MutableDouble(proto.getMax(i)));
            minMap.put(lbl,new MutableDouble(proto.getMin(i)));
            meanMap.put(lbl,new MutableDouble(proto.getMean(i)));
            sumSquaresMap.put(lbl,new MutableDouble(proto.getSumSquares(i)));
        }
        return new ImmutableRegressionInfo(countMap,labelIDMap,maxMap,minMap,meanMap,sumSquaresMap,proto.getUnknownCount(),proto.getOverallCount());
    }

    @Override
    public OutputDomainProto serialize() {
        OutputDomainProto.Builder outputBuilder = OutputDomainProto.newBuilder();

        outputBuilder.setClassName(ImmutableRegressionInfo.class.getName());
        outputBuilder.setVersion(0);

        ImmutableRegressionInfoProto.Builder data = ImmutableRegressionInfoProto.newBuilder();
        for (Map.Entry<String, MutableLong> e : countMap.entrySet()) {
            data.addLabel(e.getKey());
            data.addCount(e.getValue().longValue());
            data.addId(labelIDMap.get(e.getKey()));
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

    /**
     * Generates the domain for this regression info.
     * @param minMap The set of minimum values per dimension.
     * @return the domain, sorted lexicographically by name
     */
    private static Set<Regressor> calculateDomain(Map<String, MutableDouble> minMap) {
        TreeSet<Regressor.DimensionTuple> outputs = new TreeSet<>(Comparator.comparing(Regressor.DimensionTuple::getName));
        for (Map.Entry<String,MutableDouble> e : minMap.entrySet()) {
            outputs.add(new Regressor.DimensionTuple(e.getKey(),e.getValue().doubleValue()));
        }
        //
        // Now that we're sorted, simplify our output into a LinkedHashSet so we don't hang on to
        // the comparator we used above in the TreeSet
        LinkedHashSet<Regressor.DimensionTuple> preSortedOutputs = new LinkedHashSet<>(outputs);
        @SuppressWarnings({"unchecked","rawtypes"}) // DimensionTuple is a subtype of Regressor, and this set is immutable.
        Set<Regressor> immutableOutputs = (Set<Regressor>) (Set) Collections.unmodifiableSet(preSortedOutputs);
        return immutableOutputs;
    }

    @Override
    public Set<Regressor> getDomain() {
        return domain;
    }

    @Override
    public int getID(Regressor output) {
        return labelIDMap.getOrDefault(output.getDimensionNamesString(),-1);
    }

    @Override
    public Regressor getOutput(int id) {
        String label = idLabelMap.get(id);
        if (label != null) {
            return new Regressor(label,1.0);
        } else {
            logger.log(Level.INFO,"No entry found for id " + id);
            return null;
        }
    }

    /**
     * Gets the minimum value this RegressionInfo has seen for that id.
     * @param id The dimension to check.
     * @return The minimum value for that dimension.
     */
    public double getMin(int id) {
        return minArray[id];
    }

    /**
     * Gets the maximum value this RegressionInfo has seen for that id.
     * @param id The dimension to check.
     * @return The maximum value for that dimension.
     */
    public double getMax(int id) {
        return maxArray[id];
    }

    /**
     * Gets the mean value this RegressionInfo has seen for that id.
     * @param id The dimension to check.
     * @return The mean value for that dimension.
     */
    public double getMean(int id) {
        return meanArray[id];
    }

    /**
     * Gets the variance this RegressionInfo has seen for that id.
     * @param id The dimension to check.
     * @return The variance for that dimension.
     */
    public double getVariance(int id) {
        return varianceArray[id];
    }

    @Override
    public long getTotalObservations() {
        return overallCount;
    }

    /**
     * Computes the statistic arrays on construction and after deserialization.
     */
    private void computeStatisticArrays() {
        int size = labelIDMap.size();
        minArray = new double[size];
        maxArray = new double[size];
        meanArray = new double[size];
        varianceArray = new double[size];
        for (int i = 0; i < size; i++) {
            String name = idLabelMap.get(i);
            minArray[i] = minMap.get(name).doubleValue();
            maxArray[i] = maxMap.get(name).doubleValue();
            meanArray[i] = meanMap.get(name).doubleValue();
            varianceArray[i] = sumSquaresMap.get(name).doubleValue() / (countMap.get(name).longValue()-1);
        }
    }

    @Override
    public ImmutableRegressionInfo copy() {
        return new ImmutableRegressionInfo(this);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("RegressionInfo(");
        for (Map.Entry<String,MutableLong> e : countMap.entrySet()) {
            String name = e.getKey();
            long count = e.getValue().longValue();
            builder.append(String.format("{name=%s,id=%d,count=%d,max=%f,min=%f,mean=%f,variance=%f},",
                    name,
                    labelIDMap.get(name),
                    count,
                    maxMap.get(name).doubleValue(),
                    minMap.get(name).doubleValue(),
                    meanMap.get(name).doubleValue(),
                    (sumSquaresMap.get(name).doubleValue() / (count - 1))
            ));
        }
        builder.deleteCharAt(builder.length()-1);
        builder.append(')');
        return builder.toString();
    }

    /**
     * Returns true if the id numbers correspond to a lexicographic ordering of
     * the dimension names starting from zero, false otherwise.
     * @return True if the id mapping corresponds to lexicographic ordering of the names.
     */
    public boolean validateMapping() {
        String[] names = new String[idLabelMap.size()];
        for (Map.Entry<Integer,String> e : idLabelMap.entrySet()) {
            names[e.getKey()] = e.getValue();
        }

        String[] sortedNames = Arrays.copyOf(names,names.length);
        Arrays.sort(sortedNames);
        return Arrays.equals(names,sortedNames);
    }

    /**
     * Computes the mapping between ID numbers and regressor dimension indices.
     * <p>
     * In some situations the regressor dimension ID numbers may not use the natural ordering (i.e., the
     * lexicographic order of the dimension names).
     * This method computes the mapping from the id numbers to the natural ordering.
     * @return An array where arr[id] = natural_idx.
     */
    public int[] getIDtoNaturalOrderMapping() {
        int[] mapping = new int[idLabelMap.size()];

        SortedMap<String,Integer> sortedMap = new TreeMap<>(String::compareTo);
        sortedMap.putAll(labelIDMap);

        int i = 0;
        for (Map.Entry<String,Integer> e : sortedMap.entrySet()) {
            mapping[e.getValue()] = i;
            i++;
        }

        return mapping;
    }

    /**
     * Computes the mapping between regressor dimension indices and ID numbers.
     * <p>
     * In some situations the regressor dimension ID numbers may not use the natural ordering (i.e., the
     * lexicographic order of the dimension names).
     * This method computes the mapping from the natural ordering to the id numbers.
     * @return An array where arr[natural_idx] = id.
     */
    public int[] getNaturalOrderToIDMapping() {
        int[] mapping = new int[idLabelMap.size()];

        SortedMap<String,Integer> sortedMap = new TreeMap<>(String::compareTo);
        sortedMap.putAll(labelIDMap);

        int i = 0;
        for (Map.Entry<String,Integer> e : sortedMap.entrySet()) {
            mapping[i] = e.getValue();
            i++;
        }

        return mapping;
    }

    @Override
    public String toReadableString() {
        return toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ImmutableRegressionInfo that = (ImmutableRegressionInfo) o;
        if (unknownCount == that.unknownCount && overallCount == that.overallCount && labelIDMap.equals(that.labelIDMap)) {
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
        return Objects.hash(countMap, labelIDMap, maxMap, minMap, meanMap, sumSquaresMap, unknownCount, overallCount);
    }

    @Override
    public Iterator<Pair<Integer, Regressor>> iterator() {
        return new ImmutableInfoIterator(idLabelMap);
    }

    @Override
    public boolean domainAndIDEquals(ImmutableOutputInfo<Regressor> other) {
        if (size() == other.size()) {
            for (Map.Entry<Integer,String> e : idLabelMap.entrySet()) {
                Regressor otherReg = other.getOutput(e.getKey());
                if (otherReg == null) {
                    return false;
                } else if (!otherReg.getDimensionNamesString().equals(e.getValue())) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    private static class ImmutableInfoIterator implements Iterator<Pair<Integer, Regressor>> {

        private final Iterator<Map.Entry<Integer,String>> itr;

        public ImmutableInfoIterator(Map<Integer,String> idLabelMap) {
            itr = idLabelMap.entrySet().iterator();
        }

        @Override
        public boolean hasNext() {
            return itr.hasNext();
        }

        @Override
        public Pair<Integer, Regressor> next() {
            Map.Entry<Integer,String> e = itr.next();
            return new Pair<>(e.getKey(),new Regressor.DimensionTuple(e.getValue(),1.0));
        }
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();

        // Rebuild statistic arrays
        computeStatisticArrays();
    }
}
