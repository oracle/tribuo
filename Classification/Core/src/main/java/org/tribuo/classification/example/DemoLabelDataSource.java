/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.example;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.Example;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.provenance.ConfiguredDataSourceProvenance;
import org.tribuo.provenance.DataSourceProvenance;

import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * The base class for the 2d binary classification data sources in {@link org.tribuo.classification.example}.
 * <p>
 * The feature names are {@link #X1} and {@link #X2} and the labels are {@link #FIRST_CLASS} and {@link #SECOND_CLASS}.
 * <p>
 * Likely to be sealed to the classes in this package when we adopt Java 17.
 */
public abstract class DemoLabelDataSource implements ConfigurableDataSource<Label> {

    protected static final LabelFactory factory = new LabelFactory();

    /**
     * The first feature name.
     */
    public static final String X1 = "X1";
    /**
     * The second feature name.
     */
    public static final String X2 = "X2";

    /**
     * The feature names array.
     */
    static final String[] FEATURE_NAMES = new String[]{X1,X2};

    /**
     * The first class.
     */
    public static final Label FIRST_CLASS = new Label("X");
    /**
     * The second class.
     */
    public static final Label SECOND_CLASS = new Label("O");

    @Config(mandatory = true, description = "Number of samples to generate.")
    protected int numSamples;

    @Config(mandatory = true, description = "RNG seed.")
    protected long seed;

    // Uses java.util.Random as SplittableRandom is missing nextGaussian in versions before 17.
    protected Random rng;

    protected List<Example<Label>> examples;

    /**
     * For OLCUT.
     */
    DemoLabelDataSource() {}

    /**
     * Stores the numSamples and the seed.
     * <p>
     * Note does not call {@link #postConfig} to generate the examples,
     * this must be called by the subclass's constructor.
     * @param numSamples The number of samples to generate.
     * @param seed The RNG seed.
     */
    DemoLabelDataSource(int numSamples, long seed) {
        this.numSamples = numSamples;
        this.seed = seed;
    }

    /**
     * Configures the class. Should be called in sub-classes' postConfigs
     * after they've validated their parameters.
     */
    @Override
    public void postConfig() {
        if (numSamples < 1) {
            throw new PropertyException("","numSamples","Number of samples must be positive, found " + numSamples);
        }
        this.rng = new Random(seed);
        this.examples = Collections.unmodifiableList(generate());
    }

    /**
     * Generates the examples using the configured fields.
     * <p>
     * Is called internally by {@link #postConfig}.
     * @return The generated examples.
     */
    protected abstract List<Example<Label>> generate();

    @Override
    public LabelFactory getOutputFactory() {
        return factory;
    }

    @Override
    public DataSourceProvenance getProvenance() {
        return new DemoLabelDataSourceProvenance(this);
    }

    @Override
    public Iterator<Example<Label>> iterator() {
        return examples.iterator();
    }

    /**
     * Provenance for {@link DemoLabelDataSource}.
     */
    public static final class DemoLabelDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements ConfiguredDataSourceProvenance {
        private static final long serialVersionUID = 1L;

        /**
         * Constructs a provenance from the host data source.
         *
         * @param host The host to read.
         */
        DemoLabelDataSourceProvenance(DemoLabelDataSource host) {
            super(host, "DataSource");
        }

        /**
         * Constructs a provenance from the marshalled form.
         *
         * @param map The map of field values.
         */
        public DemoLabelDataSourceProvenance(Map<String, Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private DemoLabelDataSourceProvenance(SkeletalConfiguredObjectProvenance.ExtractedInfo info) {
            super(info);
        }

        /**
         * Extracts the relevant provenance information fields for this class.
         *
         * @param map The map to remove values from.
         * @return The extracted information.
         */
        static ExtractedInfo extractProvenanceInfo(Map<String, Provenance> map) {
            Map<String, Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters, CLASS_NAME, StringProvenance.class, DemoLabelDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters, HOST_SHORT_NAME, StringProvenance.class, DemoLabelDataSourceProvenance.class.getSimpleName()).getValue();

            return new ExtractedInfo(className, hostTypeStringName, configuredParameters, Collections.emptyMap());
        }
    }
}
