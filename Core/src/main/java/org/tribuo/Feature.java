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
import java.util.Comparator;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A class for features. Features are an immutable tuple of name and a double value.
 * <p>
 * Features can be manufactured by the {@link Example} and are not expected
 * to be long lived objects. They may be deconstructed when stored in an Example.
 * One day they should become value/inline types.
 */
public class Feature implements Serializable, Cloneable, Comparable<Feature> {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(Feature.class.getName());

    /**
     * The feature name.
     */
    protected final String name;

    /**
     * The feature value.
     */
    protected final double value;

    /**
     * Creates an immutable feature.
     * @param name The feature name.
     * @param value The feature value.
     */
    public Feature(String name, double value) {
        this.name = name;
        this.value = value;
    }

    /**
     * Returns the feature name.
     * @return The feature name
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the feature value.
     * @return The feature value.
     */
    public double getValue() {
        return value;
    }

    @Override
    public String toString() {
        return String.format("(%s, %f)", name, value);
    }

    /**
     * Returns the feature name formatted as a table cell.
     * @return The feature name.
     */
    public String toHTML() {
        String cleanName = getName().replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;");

        return String.format("<td style=\"text-align:left\">%s</td>", cleanName);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Feature feature = (Feature) o;

        if (Double.compare(feature.value, value) != 0) return false;
        return name != null ? name.equals(feature.name) : feature.name == null;
    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = name != null ? name.hashCode() : 0;
        temp = Double.doubleToLongBits(value);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    @Override
    public Feature clone() {
        try {
            return (Feature) super.clone();
        } catch (CloneNotSupportedException e) {
            logger.log(Level.SEVERE, "Clone failed, returning copy");
            return new Feature(name,value);
        }
    }

    /**
     * A comparator using the lexicographic ordering of feature names.
     * @return A lexicographic comparator.
     */
    public static Comparator<Feature> featureNameComparator() {
        return Comparator.comparing(a -> a.name);
    }

    @Override
    public int compareTo(Feature o) {
        return name.compareTo(o.name);
    }
}
