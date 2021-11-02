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

package org.tribuo.regression.example;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.OutputFactory;
import org.tribuo.Trainer;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.ConfiguredDataSourceProvenance;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Generates a single dimensional output drawn from
 * N(w_0*x_0 + w_1*x_1 + w_2*x_1*x_0 + w_3*x_1*x_1*x_1 + intercept,variance).
 * <p>
 * The features are drawn from a uniform distribution over the range.
 */
public class NonlinearGaussianDataSource implements ConfigurableDataSource<Regressor> {
    @Config(mandatory=true,description = "The number of samples to draw.")
    private int numSamples;

    @Config(description = "The feature weights. Must be a 4 element array.")
    private float[] weights = new float[]{1.0f,1.0f,1.0f,1.0f};

    @Config(description="The y-intercept of the line.")
    private float intercept = 0.0f;

    @Config(description="The variance of the noise gaussian.")
    private float variance = 1.0f;

    @Config(description = "The minimum value of x_0.")
    private float xZeroMin = -2.0f;

    @Config(description = "The maximum value of x_0.")
    private float xZeroMax = 2.0f;

    @Config(description = "The minimum value of x_1.")
    private float xOneMin = -2.0f;

    @Config(description = "The maximum value of x_1.")
    private float xOneMax = 2.0f;

    @Config(description="The RNG seed.")
    private long seed = Trainer.DEFAULT_SEED;

    private List<Example<Regressor>> examples;

    private final RegressionFactory factory = new RegressionFactory();

    private static final String[] featureNames = new String[]{"X_0","X_1"};

    /**
     * For OLCUT
     */
    private NonlinearGaussianDataSource() {}

    /**
     * Generates a single dimensional output drawn from
     * N(w_0*x_0 + w_1*x_1 + w_2*x_1*x_0 + w_3*x_1*x_1*x_1 + intercept,variance).
     * <p>
     * The weights are {@code [1,1,1,1]}, the intercept is 0, and the variance is 1.0.
     * The features are drawn from a uniform distribution over the range {@code [-2,2]}.
     * @param numSamples The size of the created datasource.
     * @param seed The rng seed to use.
     */
    public NonlinearGaussianDataSource(int numSamples, long seed) {
        if (numSamples < 0) {
            throw new IllegalArgumentException("Invalid number of sample specified, must be a positive integer, found " + numSamples);
        }
        this.numSamples = numSamples;
        this.seed = seed;
        postConfig();
    }

    /**
     * Generates a single dimensional output drawn from
     * N(w_0*x_0 + w_1*x_1 + w_2*x_1*x_0 + w_3*x_1*x_1*x_1 + intercept,variance).
     * <p>
     * The features are drawn from a uniform distribution over the range.
     * @param numSamples The size of the created datasource.
     * @param weights The feature weights.
     * @param intercept The y intercept of the line.
     * @param variance The variance of the gaussian.
     * @param xZeroMin The minimum x_0 value (inclusive).
     * @param xZeroMax The maximum x_0 value (exclusive).
     * @param xOneMin The minimum x_1 value (inclusive).
     * @param xOneMax The maximum x_1 value (exclusive).
     * @param seed The rng seed to use.
     */
    public NonlinearGaussianDataSource(int numSamples, float[] weights, float intercept, float variance,
                                       float xZeroMin, float xZeroMax, float xOneMin, float xOneMax,
                                       long seed) {
        this.numSamples = numSamples;
        this.weights = weights;
        this.intercept = intercept;
        this.variance = variance;
        this.xZeroMin = xZeroMin;
        this.xZeroMax = xZeroMax;
        this.xOneMin = xOneMin;
        this.xOneMax = xOneMax;
        this.seed = seed;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        // We use java.util.Random here because SplittableRandom doesn't have nextGaussian yet.
        Random rng = new Random(seed);
        if (weights.length != 4) {
            throw new PropertyException("","weights","Must supply 4 weights, found " + weights.length);
        }
        if (xZeroMax <= xZeroMin) {
            throw new PropertyException("","xZeroMax","xZeroMax must be greater than xZeroMin, found xZeroMax = " + xZeroMax + ", xZeroMin = " + xZeroMin);
        }
        if (xOneMax <= xOneMin) {
            throw new PropertyException("","xOneMax","xOneMax must be greater than xOneMin, found xOneMax = " + xOneMax + ", xOneMin = " + xOneMin);
        }
        if (variance <= 0.0) {
            throw new PropertyException("","variance","Variance must be positive, found variance = " + variance);
        }
        List<Example<Regressor>> examples = new ArrayList<>(numSamples);
        double zeroRange = xZeroMax - xZeroMin;
        double oneRange = xOneMax - xOneMin;
        for (int i = 0; i < numSamples; i++) {
            double xZero = (rng.nextDouble() * zeroRange) + xZeroMin;
            double xOne = (rng.nextDouble() * oneRange) + xOneMin;
            // N(w_0*x_0 + w_1*x_1 + w_2*x_1*x_0 + w_3*x_1*x_1*x_1 + intercept,variance).
            double outputValue = (weights[0] * xZero) + (weights[1]*xOne) + (weights[2]*xZero*xOne) + (weights[3]*Math.pow(xOne,3)) + intercept;
            Regressor output = new Regressor("Y",(rng.nextGaussian() * variance) + outputValue);
            ArrayExample<Regressor> e = new ArrayExample<>(output,featureNames,new double[]{xZero,xOne});
            examples.add(e);
        }
        this.examples = Collections.unmodifiableList(examples);
    }

    @Override
    public OutputFactory<Regressor> getOutputFactory() {
        return factory;
    }

    @Override
    public DataSourceProvenance getProvenance() {
        return new NonlinearGaussianDataSourceProvenance(this);
    }

    @Override
    public Iterator<Example<Regressor>> iterator() {
        return examples.iterator();
    }

    /**
     * Generates a single dimensional output drawn from
     * N(w_0*x_0 + w_1*x_1 + w_2*x_1*x_0 + w_3*x_1*x_1*x_1 + intercept,variance).
     * <p>
     * The features are drawn from a uniform distribution over the range.
     * @param numSamples The size of the output dataset.
     * @param weights The feature weights.
     * @param intercept The y intercept of the line.
     * @param variance The variance of the gaussian.
     * @param xZeroMin The minimum x_0 value (inclusive).
     * @param xZeroMax The maximum x_0 value (exclusive).
     * @param xOneMin The minimum x_1 value (inclusive).
     * @param xOneMax The maximum x_1 value (exclusive).
     * @param seed The rng seed to use.
     * @return A dataset drawn from a gaussian.
     */
    public static MutableDataset<Regressor> generateDataset(int numSamples, float[] weights, float intercept, float variance,
                                                     float xZeroMin, float xZeroMax, float xOneMin, float xOneMax,
                                                     long seed) {
        NonlinearGaussianDataSource source = new NonlinearGaussianDataSource(numSamples,weights,intercept,variance,
                xZeroMin,xZeroMax,xOneMin,xOneMax,seed);
        return new MutableDataset<>(source);
    }

    /**
     * Provenance for {@link NonlinearGaussianDataSource}.
     */
    public static class NonlinearGaussianDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements ConfiguredDataSourceProvenance {
        private static final long serialVersionUID = 1L;

        /**
         * Constructs a provenance from the host data source.
         * @param host The host to read.
         */
        NonlinearGaussianDataSourceProvenance(NonlinearGaussianDataSource host) {
            super(host,"DataSource");
        }

        /**
         * Constructs a provenance from the marshalled form.
         * @param map The map of field values.
         */
        public NonlinearGaussianDataSourceProvenance(Map<String, Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private NonlinearGaussianDataSourceProvenance(ExtractedInfo info) {
            super(info);
        }

        /**
         * Extracts the relevant provenance information fields for this class.
         * @param map The map to remove values from.
         * @return The extracted information.
         */
        protected static ExtractedInfo extractProvenanceInfo(Map<String,Provenance> map) {
            Map<String,Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters,CLASS_NAME, StringProvenance.class, NonlinearGaussianDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters,HOST_SHORT_NAME, StringProvenance.class, NonlinearGaussianDataSourceProvenance.class.getSimpleName()).getValue();

            return new ExtractedInfo(className,hostTypeStringName,configuredParameters,Collections.emptyMap());
        }
    }
}
