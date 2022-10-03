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

package org.tribuo.anomaly.example;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.MutableDataset;
import org.tribuo.OutputFactory;
import org.tribuo.Trainer;
import org.tribuo.anomaly.AnomalyFactory;
import org.tribuo.anomaly.Event;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.ConfiguredDataSourceProvenance;
import org.tribuo.provenance.DataSourceProvenance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.tribuo.anomaly.AnomalyFactory.ANOMALOUS_EVENT;
import static org.tribuo.anomaly.AnomalyFactory.EXPECTED_EVENT;

/**
 * Generates an anomaly detection dataset sampling each feature uniformly from a univariate Gaussian.
 * <p>
 * Or equivalently sampling all the features from a spherical Gaussian.
 * Can accept at most 26 features.
 * <p>
 * By default the expected means are (1.0, 2.0, 1.0, 2.0, 5.0), with variances
 * (1.0, 0.5, 0.25, 1.0, 0.1).
 * The anomalous means are (-2.0, 2.0, -2.0, 2.0, -10.0), with variances (1.0, 0.5, 0.25, 1.0, 0.1)
 * which are the same as the default expected variances.
 */
public final class GaussianAnomalyDataSource implements ConfigurableDataSource<Event> {

    private static final AnomalyFactory factory = new AnomalyFactory();

    private static final String[] allFeatureNames = new String[]{
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    };

    @Config(mandatory = true, description = "The number of samples to draw.")
    private int numSamples;

    @Config(description = "Means of the expected events.")
    private double[] expectedMeans = new double[]{1.0, 2.0, 1.0, 2.0, 5.0};

    @Config(description = "Variances of the expected events.")
    private double[] expectedVariances = new double[]{1.0, 0.5, 0.25, 1.0, 0.1};

    @Config(description = "Means of the anomalous events.")
    private double[] anomalousMeans = new double[]{-2.0, 2.0, -2.0, 2.0, -10.0};

    @Config(description = "Variances of the anomalous events.")
    private double[] anomalousVariances = new double[]{1.0, 0.5, 0.25, 1.0, 0.1};

    @Config(description = "The RNG seed.")
    private long seed = Trainer.DEFAULT_SEED;

    @Config(mandatory = true, description = "The fraction of anomalous events.")
    private float fractionAnomalous = 0.3f;

    private List<Example<Event>> examples;

    /**
     * For OLCUT.
     */
    private GaussianAnomalyDataSource() { }

    /**
     * Generates anomaly detection examples sampling each feature uniformly from a univariate Gaussian.
     * <p>
     * Or equivalently sampling all the features from a spherical Gaussian.
     * <p>
     * Can accept at most 26 features.
     *
     * @param numSamples        The size of the output dataset.
     * @param fractionAnomalous The fraction of anomalies in the generated data.
     * @param seed              The rng seed to use.
     */
    public GaussianAnomalyDataSource(int numSamples, float fractionAnomalous, long seed) {
        this.numSamples = numSamples;
        this.fractionAnomalous = fractionAnomalous;
        this.seed = seed;
        postConfig();
    }

    /**
     * Generates anomaly detection examples sampling each feature uniformly from a univariate Gaussian.
     * <p>
     * Or equivalently sampling all the features from a spherical Gaussian.
     * <p>
     * Can accept at most 26 features.
     *
     * @param numSamples         The size of the output dataset.
     * @param expectedMeans      The means of the expected event features.
     * @param expectedVariances  The variances of the expected event features.
     * @param anomalousMeans     The means of the anomalous event features.
     * @param anomalousVariances The variances of the anomalous event features.
     * @param fractionAnomalous  The fraction of anomalies to generate.
     * @param seed               The rng seed to use.
     */
    public GaussianAnomalyDataSource(int numSamples, double[] expectedMeans, double[] expectedVariances,
                                     double[] anomalousMeans, double[] anomalousVariances,
                                     float fractionAnomalous, long seed) {
        this.numSamples = numSamples;
        this.expectedMeans = expectedMeans;
        this.expectedVariances = expectedVariances;
        this.anomalousMeans = anomalousMeans;
        this.anomalousVariances = anomalousVariances;
        this.fractionAnomalous = fractionAnomalous;
        this.seed = seed;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (numSamples < 1) {
            throw new PropertyException("", "numSamples", "numSamples must be positive, found " + numSamples);
        }
        if ((expectedMeans.length > allFeatureNames.length) || (expectedMeans.length == 0)) {
            throw new PropertyException("", "expectedMeans", "Must have 1-26 features, found " + expectedMeans.length);
        }
        if (expectedMeans.length != expectedVariances.length) {
            throw new PropertyException("", "expectedMeans", "Must supply the same number of expected means and variances." +
                    " expectedMeans.length = " + expectedMeans.length +
                    " expectedVariances.length = " + expectedVariances.length);
        }
        if (anomalousMeans.length != anomalousVariances.length) {
            throw new PropertyException("", "anomalousMeans", "Must supply the same number of anomalous means and variances." +
                    " anomalousMeans.length = " + anomalousMeans.length +
                    " anomalousVariances.length = " + anomalousVariances.length);
        }
        if (fractionAnomalous < 0.0f || fractionAnomalous > 1.0f) {
            throw new PropertyException("", "fractionAnomalous", "fractionAnomalous must be between 0.0 and 1.0, found " + fractionAnomalous);
        }
        if ((fractionAnomalous != 0.0) && (anomalousMeans.length != expectedMeans.length)) {
            throw new PropertyException("", "anomalousMeans", "When sampling anomalous data there must be the same number " +
                    "of anomalous features as expected features. anomalousMeans.length = " + anomalousMeans.length +
                    ", expectedMeans.length = " + expectedMeans.length);

        }
        for (int i = 0; i < anomalousVariances.length; i++) {
            if (anomalousVariances[i] < 1e-10) {
                throw new PropertyException("","anomalousVariances", "Variances must be positive, found " + Arrays.toString(anomalousVariances));
            }
            if (expectedVariances[i] < 1e-10) {
                throw new PropertyException("","expectedVariances", "Variances must be positive, found " + Arrays.toString(expectedVariances));
            }
        }
        String[] featureNames = Arrays.copyOf(allFeatureNames, expectedMeans.length);
        // We use java.util.Random here because SplittableRandom doesn't have nextGaussian yet.
        // Once we adopt Java 17 we may switch to SplittableRandom.
        Random rng = new Random(seed);
        List<Example<Event>> examples = new ArrayList<>(numSamples);
        for (int i = 0; i < numSamples; i++) {
            double draw = rng.nextDouble();
            if (draw < fractionAnomalous) {
                List<Feature> featureList = generateFeatures(rng, featureNames, anomalousMeans, anomalousVariances);
                examples.add(new ArrayExample<>(ANOMALOUS_EVENT, featureList));
            } else {
                List<Feature> featureList = generateFeatures(rng, featureNames, expectedMeans, expectedVariances);
                examples.add(new ArrayExample<>(EXPECTED_EVENT, featureList));
            }
        }
        this.examples = Collections.unmodifiableList(examples);
    }

    @Override
    public OutputFactory<Event> getOutputFactory() {
        return factory;
    }

    @Override
    public DataSourceProvenance getProvenance() {
        return new GaussianAnomalyDataSourceProvenance(this);
    }

    @Override
    public Iterator<Example<Event>> iterator() {
        return examples.iterator();
    }

    /**
     * Generates the features based on the RNG, the means and the variances.
     *
     * @param rng       The RNG to use.
     * @param names     The feature names.
     * @param means     The feature means.
     * @param variances The feature variances.
     * @return A sampled feature list.
     */
    private static List<Feature> generateFeatures(Random rng, String[] names, double[] means, double[] variances) {
        if ((names.length != means.length) || (names.length != variances.length)) {
            throw new IllegalArgumentException("Names, means and variances must be the same length");
        }

        List<Feature> features = new ArrayList<>();

        for (int i = 0; i < names.length; i++) {
            double value = (rng.nextGaussian() * Math.sqrt(variances[i])) + means[i];
            features.add(new Feature(names[i], value));
        }

        return features;
    }

    /**
     * Generates an anomaly detection dataset sampling each feature uniformly from a univariate Gaussian.
     * <p>
     * Or equivalently sampling all the features from a spherical Gaussian.
     * <p>
     * Can accept at most 26 features.
     *
     * @param numSamples         The size of the output dataset.
     * @param expectedMeans      The means of the expected event features.
     * @param expectedVariances  The variances of the expected event features.
     * @param anomalousMeans     The means of the anomalous event features.
     * @param anomalousVariances The variances of the anomalous event features.
     * @param fractionAnomalous  The fraction of anomalies to generate.
     * @param seed               The rng seed to use.
     * @return A dataset drawn from a gaussian.
     */
    public static MutableDataset<Event> generateDataset(int numSamples, double[] expectedMeans, double[] expectedVariances,
                                                 double[] anomalousMeans, double[] anomalousVariances,
                                                 float fractionAnomalous, long seed) {
        GaussianAnomalyDataSource source = new GaussianAnomalyDataSource(numSamples, expectedMeans, expectedVariances, anomalousMeans, anomalousVariances, fractionAnomalous, seed);
        return new MutableDataset<>(source);
    }

    /**
     * Provenance for {@link GaussianAnomalyDataSource}.
     */
    public static final class GaussianAnomalyDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements ConfiguredDataSourceProvenance {
        private static final long serialVersionUID = 1L;

        /**
         * Constructs a provenance from the host data source.
         *
         * @param host The host to read.
         */
        GaussianAnomalyDataSourceProvenance(GaussianAnomalyDataSource host) {
            super(host, "DataSource");
        }

        /**
         * Constructs a provenance from the marshalled form.
         *
         * @param map The map of field values.
         */
        public GaussianAnomalyDataSourceProvenance(Map<String, Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private GaussianAnomalyDataSourceProvenance(SkeletalConfiguredObjectProvenance.ExtractedInfo info) {
            super(info);
        }

        /**
         * Extracts the relevant provenance information fields for this class.
         *
         * @param map The map to remove values from.
         * @return The extracted information.
         */
        protected static ExtractedInfo extractProvenanceInfo(Map<String, Provenance> map) {
            Map<String, Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters, CLASS_NAME, StringProvenance.class, GaussianAnomalyDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters, HOST_SHORT_NAME, StringProvenance.class, GaussianAnomalyDataSourceProvenance.class.getSimpleName()).getValue();

            return new ExtractedInfo(className, hostTypeStringName, configuredParameters, Collections.emptyMap());
        }
    }
}
