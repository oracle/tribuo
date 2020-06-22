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

package org.tribuo.regression.example;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.Dataset;
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
 * Generates a single dimensional output drawn from N(slope*x + intercept,variance).
 * <p>
 * The single feature is drawn from a uniform distribution over the range.
 * <p>
 * Set slope to zero to draw from a gaussian.
 */
public class GaussianDataSource implements ConfigurableDataSource<Regressor> {
    @Config(mandatory=true)
    private int numSamples;

    @Config
    private float slope;

    @Config
    private float intercept;

    @Config
    private float variance = 1.0f;

    @Config(mandatory=true)
    private float xMin;

    @Config(mandatory=true)
    private float xMax;

    @Config
    private long seed = Trainer.DEFAULT_SEED;

    private List<Example<Regressor>> examples;

    private final RegressionFactory factory = new RegressionFactory();

    /**
     * For OLCUT
     */
    private GaussianDataSource() {}

    /**
     * Generates a single dimensional output drawn from N(slope*x + intercept,variance).
     * <p>
     * The single feature is drawn from a uniform distribution over the range.
     * <p>
     * Set slope to zero to draw from a gaussian.
     * @param numSamples The size of the output dataset.
     * @param slope The slope of the line.
     * @param intercept The y intercept of the line.
     * @param variance The variance of the gaussian.
     * @param xMin The minimum x value (inclusive).
     * @param xMax The maximum x value (exclusive).
     * @param seed The rng seed to use.
     */
    public GaussianDataSource(int numSamples, float slope, float intercept, float variance, float xMin, float xMax, long seed) {
        this.numSamples = numSamples;
        this.slope = slope;
        this.intercept = intercept;
        this.variance = variance;
        this.xMin = xMin;
        this.xMax = xMax;
        this.seed = seed;
        postConfig();
    }

    @Override
    public void postConfig() {
        Random rng = new Random(seed);
        List<Example<Regressor>> examples = new ArrayList<>(numSamples);
        if (xMax <= xMin) {
            throw new PropertyException("","xMax","xMax must be greater than xMin, found xMax = " + xMax + ", xMin = " + xMin);
        }
        if (variance <= 0.0) {
            throw new PropertyException("","variance","Variance must be positive, found variance = " + variance);
        }
        double range = xMax - xMin;
        for (int i = 0; i < numSamples; i++) {
            double input = (rng.nextDouble() * range) + xMin;
            Regressor output = new Regressor("Y",(rng.nextGaussian() * variance) + ((slope * input) + intercept));
            ArrayExample<Regressor> e = new ArrayExample<>(output,new String[]{"X"},new double[]{input});
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
        return new GaussianDataSourceProvenance(this);
    }

    @Override
    public Iterator<Example<Regressor>> iterator() {
        return examples.iterator();
    }

    /**
     * Generates a single dimensional output drawn from N(slope*x + intercept,variance).
     * <p>
     * The single feature is drawn from a uniform distribution over the range.
     * <p>
     * Set slope to zero to draw from a gaussian.
     * @param numSamples The size of the output dataset.
     * @param slope The slope of the line.
     * @param intercept The y intercept of the line.
     * @param variance The variance of the gaussian.
     * @param xMin The minimum x value (inclusive).
     * @param xMax The maximum x value (exclusive).
     * @param seed The rng seed to use.
     * @return A dataset drawn from a gaussian.
     */
    public static Dataset<Regressor> generateDataset(int numSamples, float slope, float intercept, float variance, float xMin, float xMax, long seed) {
        GaussianDataSource source = new GaussianDataSource(numSamples,slope,intercept,variance,xMin,xMax,seed);
        return new MutableDataset<>(source);
    }

    public static class GaussianDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements ConfiguredDataSourceProvenance {
        private static final long serialVersionUID = 1L;

        GaussianDataSourceProvenance(GaussianDataSource host) {
            super(host,"DataSource");
        }

        public GaussianDataSourceProvenance(Map<String, Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private GaussianDataSourceProvenance(ExtractedInfo info) {
            super(info);
        }

        protected static ExtractedInfo extractProvenanceInfo(Map<String,Provenance> map) {
            Map<String,Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters,CLASS_NAME, StringProvenance.class, GaussianDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters,HOST_SHORT_NAME, StringProvenance.class, GaussianDataSourceProvenance.class.getSimpleName()).getValue();

            return new ExtractedInfo(className,hostTypeStringName,configuredParameters,Collections.emptyMap());
        }
    }
}
