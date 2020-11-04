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

/**
 * A feature map that can record new feature value observations.
 */
public class MutableFeatureMap extends FeatureMap {
    private static final long serialVersionUID = 2L;

    private final boolean convertHighCardinality;

    /**
     * Creates an empty feature map which converts high cardinality categorical variable infos into reals.
     * <p>
     * The conversion threshold is {@link CategoricalInfo#THRESHOLD}.
     */
    public MutableFeatureMap() {
        this(true);
    }

    /**
     * Creates an empty feature map which can optionally convert high cardinality categorical variable infos into reals.
     * <p>
     * The conversion threshold is {@link CategoricalInfo#THRESHOLD}.
     * @param convertHighCardinality Should this feature map convert high cardinality categorical variables into real variables?
     */
    public MutableFeatureMap(boolean convertHighCardinality) {
        super();
        this.convertHighCardinality = convertHighCardinality;
    }

    /**
     * Adds a variable info into the feature map.
     * <p>
     * Returns the old one if there was a name collision, otherwise returns null.
     * @param info The info to add.
     * @return The old variable info or null.
     */
    public VariableInfo put(VariableInfo info) {
        VariableInfo old = m.put(info.getName(), info);
        return old;
    }

    /**
     * Adds an occurrence of a feature with a given name.
     *
     * @param name the name of the feature.
     * @param value the observed value of that feature.
     */
    public void add(String name, double value) {
        SkeletalVariableInfo info = (SkeletalVariableInfo) m.computeIfAbsent(name, CategoricalInfo::new);
        info.observe(value);

        // If there are too many categories, convert into a real info and drop the old categorical info.
        if (convertHighCardinality && info instanceof CategoricalInfo) {
            CategoricalInfo cInfo = (CategoricalInfo) info;
            if (cInfo.getUniqueObservations() > CategoricalInfo.THRESHOLD) {
                m.put(name,cInfo.generateRealInfo());
            }
        }
    }

    /**
     * Clears all the feature observations.
     */
    public void clear() {
        m.clear();
    }

}
