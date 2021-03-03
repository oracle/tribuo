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
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
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

    private ImmutableRegressionInfo(ImmutableRegressionInfo info) {
        super(info);
        idLabelMap = new LinkedHashMap<>();
        idLabelMap.putAll(info.idLabelMap);
        labelIDMap = new LinkedHashMap<>();
        labelIDMap.putAll(info.labelIDMap);
        domain = calculateDomain(minMap);
    }

    ImmutableRegressionInfo(RegressionInfo info) {
        super(info);
        idLabelMap = new LinkedHashMap<>();
        labelIDMap = new LinkedHashMap<>();
        int counter = 0;
        for (Map.Entry<String,MutableLong> e : countMap.entrySet()) {
            idLabelMap.put(counter,e.getKey());
            labelIDMap.put(e.getKey(),counter);
            counter++;
        }
        domain = calculateDomain(minMap);
    }

    ImmutableRegressionInfo(RegressionInfo info, Map<Regressor,Integer> mapping) {
        super(info);
        if (mapping.size() != info.size()) {
            throw new IllegalStateException("Mapping and info come from different sources, mapping.size() = " + mapping.size() + ", info.size() = " + info.size());
        }

        idLabelMap = new HashMap<>();
        labelIDMap = new HashMap<>();
        for (Map.Entry<Regressor, Integer> e : mapping.entrySet()) {
            Regressor r = e.getKey();
            String[] names = r.getNames();
            if (names.length == 1) {
                idLabelMap.put(e.getValue(), names[0]);
                labelIDMap.put(names[0], e.getValue());
            } else {
                throw new IllegalArgumentException("Mapping must contain a single regression dimension per id, but contains " + Arrays.toString(names) + " -> " + e.getValue());
            }
        }
        domain = calculateDomain(minMap);
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
        LinkedHashSet<Regressor.DimensionTuple> preSortedOutputs = new LinkedHashSet<>();
        preSortedOutputs.addAll(outputs);
        @SuppressWarnings("unchecked") // DimensionTuple is a subtype of Regressor, and this set is immutable.
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

    @Override
    public long getTotalObservations() {
        return overallCount;
    }

    @Override
    public ImmutableRegressionInfo copy() {
        return new ImmutableRegressionInfo(this);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("MultipleRegressionOutput(");
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

    @Override
    public String toReadableString() {
        return toString();
    }

    @Override
    public Iterator<Pair<Integer, Regressor>> iterator() {
        return new ImmutableInfoIterator(idLabelMap);
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
            return new Pair<>(e.getKey(),new Regressor(e.getValue(),1.0));
        }
    }
}
