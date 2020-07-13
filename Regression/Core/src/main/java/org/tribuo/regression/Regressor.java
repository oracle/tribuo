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

import com.oracle.labs.mlrg.olcut.util.Pair;
import com.oracle.labs.mlrg.olcut.util.SortUtil;
import org.tribuo.Output;
import org.tribuo.OutputInfo;
import org.tribuo.util.Util;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * An {@link Output} for n-dimensional real valued regression.
 * <p>
 * In addition to the regressed values, it may optionally contain
 * variances. Otherwise the variances are set to {@link Double#NaN}.
 * </p>
 * <p>
 * Within a {@link org.tribuo.DataSource} or {@link org.tribuo.Dataset}
 * each Regressor must contain the same set of named dimensions. The dimensions stored in a
 * Regressor are sorted by the natural ordering of their names (i.e. using the String comparator).
 * This allows the use of direct indexing into the elements.
 * </p>
 * <p>
 * Note {@link Regressor#fullEquals} compares the dimensions, the regressed values and the
 * variances. However unlike {@link Double#equals}, if the two variances being compared are
 * set to the sentinel value of {@link Double#NaN}, then they are considered equal.
 * </p>
 */
public class Regressor implements Output<Regressor>, Iterable<Regressor.DimensionTuple> {
    private static final long serialVersionUID = 1L;
    public static final double TOLERANCE = 1e-12;

    public static final String DEFAULT_NAME = "DIM";

    private final String[] names;

    private final double[] values;

    private final double[] variances;

    private boolean hashCache = false;

    private int hashCode;

    /**
     * Constructs a regressor from the supplied named values. Throws {@link IllegalArgumentException}
     * if the arrays are not all the same size.
     * @param names The names of the dimensions.
     * @param values The values of the dimensions.
     * @param variances The variances of the specified values.
     */
    public Regressor(String[] names, double[] values, double[] variances) {
        if ((names.length != values.length) || (names.length != variances.length)) {
            throw new IllegalArgumentException("Arrays must be the same length, names.length="+names.length+", values.length="+values.length+",variances.length="+variances.length);
        }
        int[] indices = SortUtil.argsort(names,true);
        this.names = new String[names.length];
        this.values = new double[values.length];
        this.variances = new double[variances.length];
        for (int i = 0; i < indices.length; i++) {
            this.names[i] = names[indices[i]];
            this.values[i] = values[indices[i]];
            this.variances[i] = variances[indices[i]];
        }
        Set<String> nameSet = new HashSet<>(Arrays.asList(this.names));
        if (nameSet.size() != this.names.length) {
            throw new IllegalArgumentException("Names must all be unique, found " + (this.names.length - nameSet.size()) + " duplicates");
        }
    }

    /**
     * Constructs a regressor from the supplied named values. Uses {@link Double#NaN} as
     * the variances.
     * @param names The names of the dimensions.
     * @param values The values of the dimensions.
     */
    public Regressor(String[] names, double[] values) {
        this(names, values, Util.generateUniformVector(values.length,Double.NaN));
    }

    /**
     * Constructs a regressor from the supplied dimension tuples.
     * @param dimensions The named values to use.
     */
    public Regressor(DimensionTuple[] dimensions) {
        int[] indices = SortUtil.argsort(extractNames(dimensions),true);
        this.names = new String[dimensions.length];
        this.values = new double[names.length];
        this.variances = new double[names.length];
        for (int i = 0; i < dimensions.length; i++) {
            DimensionTuple cur = dimensions[indices[i]];
            names[i] = cur.getName();
            values[i] = cur.getValue();
            variances[i] = cur.getVariance();
        }
        Set<String> nameSet = new HashSet<>(Arrays.asList(this.names));
        if (nameSet.size() != this.names.length) {
            throw new IllegalArgumentException("Names must be unique, found " + (this.names.length - nameSet.size()) + " duplicates");
        }
    }

    /**
     * Constructs a regressor containing a single dimension, using
     * {@link Double#NaN} as the variance.
     * @param name The name of the dimension.
     * @param value The value of the dimension.
     */
    public Regressor(String name, double value) {
        this(name,value,Double.NaN);
    }

    /**
     * Constructs a regressor containing a single dimension.
     * @param name The name of the dimension.
     * @param value The value of the dimension.
     * @param variance The variance of this value.
     */
    public Regressor(String name, double value, double variance) {
        this.names = new String[]{name};
        this.values = new double[]{value};
        this.variances = new double[]{variance};
    }

    /**
     * Returns the number of dimensions in this regressor.
     * @return The number of dimensions.
     */
    public int size() {
        return names.length;
    }

    /**
     * The names of the dimensions. Always sorted by their natural ordering.
     * @return The names of the dimensions.
     */
    public String[] getNames() {
        return names;
    }

    /**
     * Returns the regression values.
     * @return The regression values.
     */
    public double[] getValues() {
        return values;
    }

    /**
     * The variances of the regressed values, if known.
     *
     * Returns Double.NaN otherwise.
     * @return The variance of the regressed values.
     */
    public double[] getVariances() {
        return variances;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        for (int i = 0; i < names.length; i++) {
            builder.append('(');
            if (Double.isNaN(variances[i])) {
                builder.append(names[i]);
                builder.append(',');
                builder.append(values[i]);
            } else {
                builder.append(names[i]);
                builder.append(',');
                builder.append(values[i]);
                builder.append(",var=");
                builder.append(variances[i]);
            }
            builder.append("),");
        }

        builder.deleteCharAt(builder.length()-1);

        return builder.toString();
    }

    /**
     *  Returns a dimension tuple for the requested dimension, or optional empty if
     *  it's not valid.
     * @param name The dimension name.
     * @return A tuple representing that dimension.
     */
    public Optional<DimensionTuple> getDimension(String name) {
        int i = 0;
        while (i < names.length) {
            if (names[i].equals(name)) {
                return Optional.of(new DimensionTuple(name, values[i], variances[i]));
            }
            i++;
        }
        return Optional.empty();
    }

    @Override
    public Iterator<DimensionTuple> iterator() {
        return new RegressorIterator();
    }

    @Override
    public Regressor copy() {
        return new Regressor(names,Arrays.copyOf(values,values.length),Arrays.copyOf(variances,variances.length));
    }

    @Override
    public String getSerializableForm(boolean includeConfidence) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < names.length; i++) {
            builder.append(names[i]);
            builder.append('=');
            builder.append(values[i]);
            if (includeConfidence && !Double.isNaN(variances[i])) {
                builder.append('\u00B1');
                builder.append(variances[i]);
            }
            builder.append(',');
        }
        builder.deleteCharAt(builder.length()-1);
        return builder.toString();
    }

    @Override
    public boolean fullEquals(Regressor other) {
        if (!Arrays.equals(names,other.names)) {
            return false;
        } else {
            for (int i = 0; i < values.length; i++) {
                if (Math.abs(values[i] - other.values[i]) > TOLERANCE) {
                    return false;
                } else {
                    double ourVar = variances[i];
                    double otherVar = other.variances[i];
                    if ((Math.abs(ourVar-otherVar) > TOLERANCE) || (Double.isNaN(ourVar) ^ Double.isNaN(otherVar))) {
                        return false;
                    }
                }
            }
            return true;
        }
    }

    /**
     * Regressors are equal if they have the same number of dimensions and equal dimension names.
     *
     * @param o An object.
     * @return True if Object is a Regressor with the same dimension names, false otherwise.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        } else if (o instanceof Regressor) {
            return Arrays.deepEquals(names,((Regressor)o).names);
        } else {
            return false;
        }
    }

    /**
     * Regressor's hashcode is based on the hash of the dimension names.
     * <p>
     * It's cached on first access.
     * @return A hashcode.
     */
    @Override
    public synchronized int hashCode() {
        if (!hashCache) {
            hashCode = 11;
            for (int i = 0; i < names.length; i++) {
                hashCode ^= names[i].hashCode();
            }
            hashCache = true;
        }
        return hashCode;
    }

    /**
     * Returns a comma separated list of the dimension names.
     * @return The dimension names comma separated.
     */
    public String getDimensionNamesString() {
        return getDimensionNamesString(',');
    }

    /**
     * Returns a delimiter separated list of the dimension names.
     * @param separator The separator to use.
     * @return The dimension names.
     */
    public String getDimensionNamesString(char separator) {
        return String.join(""+separator,names);
    }

    /**
     * Extracts a String array of each dimension name from an array of DimensionTuples.
     * @param values The dimensions.
     * @return The names of the dimensions.
     */
    private static String[] extractNames(DimensionTuple[] values) {
        String[] extractedNames = new String[values.length];

        for (int i = 0; i < values.length; i++) {
            extractedNames[i] = values[i].getName();
        }

        return extractedNames;
    }

    /**
     * Extracts the names from the supplied Regressor domain in their canonical order.
     * @param info The OutputInfo to use.
     * @return The dimension names from this domain.
     */
    public static String[] extractNames(OutputInfo<Regressor> info) {
        String[] extractedNames = new String[info.size()];
        int i = 0;
        for (Regressor r : info.getDomain()) {
            extractedNames[i] = r.getNames()[0];
            i++;
        }
        Arrays.sort(extractedNames);
        return extractedNames;
    }

    /**
     * Parses a string of the form:
     * <pre>
     * dimension-name=output,...,dimension-name=output
     * </pre>
     * where output must be readable by {@link Double#parseDouble}.
     * @param s The string form of a multiple regressor.
     * @return A regressor parsed from the input string.
     */
    public static Regressor parseString(String s) {
        return parseString(s,',');
    }

    /**
     * Parses a string of the form:
     * <pre>
     * dimension-name=output&lt;splitChar&gt;...&lt;splitChar&gt;dimension-name=output
     * </pre>
     * where output must be readable by {@link Double#parseDouble}.
     * @param s The string form of a regressor.
     * @param splitChar The char to split on.
     * @return A regressor parsed from the input string.
     */
    public static Regressor parseString(String s, char splitChar) {
        if (splitChar == '=') {
            throw new IllegalArgumentException("Can't split on an equals symbol");
        }
        String[] tokens = s.split(""+splitChar);

        String[] names = new String[tokens.length];
        double[] values = new double[tokens.length];

        Set<String> nameSet = new HashSet<>();

        for (int i = 0; i < tokens.length; i++) {
            Pair<String,Double> element = parseElement(i,tokens[i]);
            names[i] = element.getA();
            values[i] = element.getB();
            nameSet.add(element.getA());
        }

        if (nameSet.size() != tokens.length) {
            throw new IllegalArgumentException("Duplicated dimension names");
        }

        return new Regressor(names,values);
    }

    /**
     * Parses a string of the form:
     * <pre>
     * dimension-name=output-double
     * </pre>
     * where the output must be readable by {@link Double#parseDouble}.
     * @param idx The index of this string in a list.
     * @param s The string form of a single dimension from a regressor.
     * @return A tuple representing the dimension name and the value.
     */
    public static Pair<String,Double> parseElement(int idx, String s) {
        String[] split = s.split("=");
        if (split.length == 2) {
            return new Pair<>(split[0], Double.parseDouble(split[1]));
        } else if (split.length == 1) {
            //No dimension name found.
            return new Pair<>(DEFAULT_NAME + "-" + idx, Double.parseDouble(split[0]));
        } else {
            throw new IllegalArgumentException("Failed to parse element " + s);
        }
    }

    /**
     * Creates a Regressor from a list of dimension tuples.
     * @param dimensions The dimensions to use.
     * @return A Regressor representing these dimensions.
     */
    public static Regressor createFromPairList(List<Pair<String,Double>> dimensions) {
        int numDimensions = dimensions.size();
        String[] names = new String[numDimensions];
        double[] values = new double[numDimensions];
        for (int i = 0; i < numDimensions; i++) {
            Pair<String,Double> p = dimensions.get(i);
            names[i] = p.getA();
            values[i] = p.getB();
        }
        return new Regressor(names,values);
    }

    public final static class DimensionTuple extends Regressor {
        private static final long serialVersionUID = 1L;

        private final String name;
        private final double value;
        private final double variance;

        public DimensionTuple(String name, double value, double variance) {
            super(name,value,variance);
            this.name = name;
            this.value = value;
            this.variance = variance;
        }

        public DimensionTuple(String name, double value) {
            this(name,value,Double.NaN);
        }

        @Override
        public int size() {
            return 1;
        }

        @Override
        public String toString() {
            if (Double.isNaN(variance)) {
                return name+"="+value;
            } else {
                return name+"=("+value+",var="+variance+")";
            }
        }

        @Override
        public Optional<DimensionTuple> getDimension(String name) {
            if (this.name.equals(name)) {
                return Optional.of(this);
            } else {
                return Optional.empty();
            }
        }

        @Override
        public Iterator<DimensionTuple> iterator() {
            return Collections.singletonList(this).iterator();
        }

        @Override
        public DimensionTuple copy() {
            return new DimensionTuple(name,value,variance);
        }

        public String getName() {
            return name;
        }

        public double getValue() {
            return value;
        }

        public double getVariance() {
            return variance;
        }

        @Override
        public String getSerializableForm(boolean includeConfidence) {
            String tmp = name + "=" + value;
            if (includeConfidence && !Double.isNaN(variance)) {
                return tmp + "\u00B1" + variance;
            } else {
                return tmp;
            }
        }

        @Override
        public boolean fullEquals(Regressor other) {
            if (!equals(other)) {
                return false;
            } else {
                // Now check values for equality
                // Must have only one value
                double otherValue = other.values[0];
                double otherVar = other.variances[0];
                if ((Math.abs(value-otherValue) > TOLERANCE) || (Double.isNaN(value) ^ Double.isNaN(otherValue))) {
                    return false;
                }
                return (!(Math.abs(variance - otherVar) > TOLERANCE)) && (Double.isNaN(variance) == Double.isNaN(otherVar));
            }
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            } else if (o instanceof DimensionTuple) {
                return name.equals(((DimensionTuple)o).name);
            } else if (o instanceof Regressor) {
                Regressor other = (Regressor) o;
                return other.size() == 1 && other.getNames()[0].equals(name);
            } else {
                return false;
            }
        }

        /**
         * All regressors have a hashcode based on only the dimension names.
         * @return A hashcode.
         */
        @Override
        public int hashCode() {
            return 11 ^ name.hashCode();
        }

        @Override
        public String getDimensionNamesString() {
            return name;
        }
    }

    private class RegressorIterator implements Iterator<DimensionTuple> {
        private int i = 0;

        @Override
        public boolean hasNext() {
            return i < names.length;
        }

        @Override
        public DimensionTuple next() {
            DimensionTuple r = new DimensionTuple(names[i],values[i],variances[i]);
            i++;
            return r;
        }
    }
}
