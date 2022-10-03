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

package org.tribuo.multilabel.example;

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
import org.tribuo.classification.Label;
import org.tribuo.impl.ArrayExample;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;
import org.tribuo.provenance.ConfiguredDataSourceProvenance;
import org.tribuo.provenance.DataSourceProvenance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Generates a multi label output drawn from a series of functions.
 * <p>
 * The functions are:
 * <ul>
 *     <li>y_0 is positive if N(w_00*x_0 + w_01*x_1 + w_02*x_1*x_0 + w_03*x_1*x_1*x_1,variance) &gt; threshold_0.</li>
 *     <li>y_1 is positive if N(w_10*x_0 + w_11*x_1 + w_12*x_1*x_0 + w_13*x_1*x_1*x_1,variance) &lt; threshold_1.</li>
 *     <li>y_2 is positive if N(w_20*x_0 + w_21*x_2 + w_22*x_1*x_0 + w_23*x_1*x_2*x_2,variance) &gt; threshold_2.</li>
 * </ul>
 * There are 4 features x_0, x_1, x_2, x_3. x_0 and x_1 are used by all three labels,
 * x_2 is only used by y_2, and x_3 is irrelevant.
 * <p>
 * By default y_1 is the inverse of y_0, and y_2 shares the same weights w_0 and w_2.
 * <ul>
 *     <li>y_0 weights = [1.0,1.0,1.0,1.0]</li>
 *     <li>y_1 weights = [1.0,1.0,1.0,1.0]</li>
 *     <li>y_2 weights = [1.0,-3.0,1.0,3.0]</li>
 *     <li>threshold = [0.0,0.0,2.0</li>
 * </ul>
 * <p>
 * The features are drawn from a uniform distribution over the range.
 */
public final class MultiLabelGaussianDataSource implements ConfigurableDataSource<MultiLabel> {
    @Config(mandatory = true, description = "The number of samples to draw.")
    private int numSamples;

    @Config(description = "The feature weights. Must be a 4 element array.")
    private float[] yZeroWeights = new float[]{1.0f, 1.0f, 1.0f, 1.0f};

    @Config(description = "The feature weights. Must be a 4 element array.")
    private float[] yOneWeights = new float[]{1.0f, 1.0f, 1.0f, 1.0f};

    @Config(description = "The feature weights. Must be a 4 element array.")
    private float[] yTwoWeights = new float[]{1.0f, -3.0f, 1.0f, 3.0f};

    @Config(description = "The threshold for each class.")
    private float[] threshold = new float[]{0.0f, 0.0f, 2.0f};

    @Config(description = "Negate the computed value before thresholding it.")
    private boolean[] negate = new boolean[]{false, true, false};

    @Config(description = "The variance of the noise gaussian.")
    private float variance = 0.1f;

    @Config(description = "The minimum values of the Xs.")
    private float[] xMin = new float[]{-2.0f, -2.0f, -2.0f, -2.0f};

    @Config(description = "The maximum values of the Xs.")
    private float[] xMax = new float[]{2.0f, 2.0f, 2.0f, 2.0f};

    @Config(description = "The RNG seed.")
    private long seed = Trainer.DEFAULT_SEED;

    private List<Example<MultiLabel>> examples;

    private final MultiLabelFactory factory = new MultiLabelFactory();

    private static final String[] FEATURE_NAMES = new String[]{"X_0", "X_1", "X_2", "X_3"};
    private static final String[] LABEL_NAMES = new String[]{"Y_0", "Y_1", "Y_2"};

    /**
     * For OLCUT.
     */
    private MultiLabelGaussianDataSource() {}

    /**
     * Generates a multi-label output drawn from three gaussian functions.
     * <ul>
     *      <li>N(w_00*x_0 + w_01*x_1 + w_02*x_1*x_0 + w_03*x_1*x_1*x_1,variance)</li>
     *      <li>N(w_10*x_0 + w_11*x_1 + w_12*x_1*x_0 + w_13*x_1*x_1*x_1,variance)</li>
     *      <li>N(w_20*x_0 + w_21*x_2 + w_22*x_1*x_0 + w_23*x_1*x_2*x_2,variance)</li>
     * </ul>
     * <p>
     * The features are drawn from a uniform distribution over the range.
     *
     * @param numSamples   The size of the output dataset.
     * @param yZeroWeights The feature weights for label y_0.
     * @param yOneWeights  The feature weights for label y_1.
     * @param yTwoWeights  The feature weights for label y_2.
     * @param threshold    The y threshold of each label.
     * @param negate       Should the computed value be negated before thresholding?
     * @param variance     The variance of the gaussian.
     * @param xMin         The minimum feature values (inclusive).
     * @param xMax         The maximum feature values (exclusive).
     * @param seed         The rng seed to use.
     */
    public MultiLabelGaussianDataSource(int numSamples, float[] yZeroWeights, float[] yOneWeights, float[] yTwoWeights,
                                        float[] threshold, boolean[] negate, float variance, float[] xMin, float[] xMax,
                                        long seed) {
        this.numSamples = numSamples;
        this.yZeroWeights = yZeroWeights;
        this.yOneWeights = yOneWeights;
        this.yTwoWeights = yTwoWeights;
        this.threshold = threshold;
        this.negate = negate;
        this.variance = variance;
        this.xMin = xMin;
        this.xMax = xMax;
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
        if (yZeroWeights.length != 4) {
            throw new PropertyException("", "yZeroWeights", "Must supply 4 yZeroWeights, found " + yZeroWeights.length);
        }
        if (yOneWeights.length != 4) {
            throw new PropertyException("", "yOneWeights", "Must supply 4 yOneWeights, found " + yOneWeights.length);
        }
        if (yTwoWeights.length != 4) {
            throw new PropertyException("", "yTwoWeights", "Must supply 4 yTwoWeights, found " + yTwoWeights.length);
        }
        if (threshold.length != 3) {
            throw new PropertyException("", "threshold", "Must supply 3 values for threshold, found " + threshold.length);
        }
        if (negate.length != 3) {
            throw new PropertyException("", "negate", "Must supply 3 values for negate, found " + negate.length);
        }
        if (xMin.length != 4) {
            throw new PropertyException("", "xMin", "Must supply 4 feature minimums, found " + xMin.length);
        }
        if (xMax.length != 4) {
            throw new PropertyException("", "xMax", "Must supply 4 feature maximums, found " + xMax.length);
        }
        float[] range = new float[4];
        for (int i = 0; i < 4; i++) {
            if (xMin[i] > xMax[i]) {
                throw new PropertyException("", "xMin", "Feature minimums must be below the maximums, found min = " + Arrays.toString(xMin) + " and max = " + Arrays.toString(xMax));
            } else {
                range[i] = xMax[i] - xMin[i];
            }
        }
        if (variance <= 0.0) {
            throw new PropertyException("", "variance", "Variance must be positive, found variance = " + variance);
        }
        List<Example<MultiLabel>> examples = new ArrayList<>(numSamples);
        for (int i = 0; i < numSamples; i++) {
            double[] features = new double[4];
            for (int j = 0; j < features.length; j++) {
                features[j] = (rng.nextDouble() * range[j]) + xMin[j];
            }
            /*
             * y_0 is positive if N(w_00*x_0 + w_01*x_1 + w_02*x_1*x_0 + w_03*x_1*x_1*x_1,variance) > threshold_0
             * y_1 is positive if N(w_10*x_0 + w_11*x_1 + w_12*x_1*x_0 + w_13*x_1*x_1*x_1,variance) < threshold_1
             * y_2 is positive if N(w_20*x_0 + w_21*x_2 + w_22*x_1*x_0 + w_23*x_1*x_2*x_2,variance) > threshold_2
             */
            double yZero = (rng.nextGaussian() * variance) + ((yZeroWeights[0] * features[0]) + (yZeroWeights[1] * features[1]) + (yZeroWeights[2] * features[0] * features[1]) + (yZeroWeights[3] * Math.pow(features[1], 3)));
            double yOne = (rng.nextGaussian() * variance) + ((yOneWeights[0] * features[0]) + (yOneWeights[1] * features[1]) + (yOneWeights[2] * features[0] * features[1]) + (yOneWeights[3] * Math.pow(features[1], 3)));
            double yTwo = (rng.nextGaussian() * variance) + ((yTwoWeights[0] * features[0]) + (yTwoWeights[1] * features[2]) + (yTwoWeights[2] * features[0] * features[1]) + (yTwoWeights[3] * features[1] * features[2] * features[2]));
            if (negate[0]) {
                yZero = -yZero;
            }
            if (negate[1]) {
                yOne = -yOne;
            }
            if (negate[2]) {
                yTwo = -yTwo;
            }
            Set<Label> labels = new HashSet<>();
            if (yZero > threshold[0]) {
                labels.add(new Label(LABEL_NAMES[0]));
            }
            if (yOne > threshold[1]) {
                labels.add(new Label(LABEL_NAMES[1]));
            }
            if (yTwo > threshold[2]) {
                labels.add(new Label(LABEL_NAMES[2]));
            }
            MultiLabel output = new MultiLabel(labels);
            ArrayExample<MultiLabel> e = new ArrayExample<>(output, FEATURE_NAMES, features);
            examples.add(e);
        }
        this.examples = Collections.unmodifiableList(examples);
    }

    @Override
    public OutputFactory<MultiLabel> getOutputFactory() {
        return factory;
    }

    @Override
    public DataSourceProvenance getProvenance() {
        return new MultiLabelGaussianDataSourceProvenance(this);
    }

    @Override
    public Iterator<Example<MultiLabel>> iterator() {
        return examples.iterator();
    }

    /**
     * Generates a multi-label output drawn from three gaussian functions.
     * <ul>
     *      <li>N(w_00*x_0 + w_01*x_1 + w_02*x_1*x_0 + w_03*x_1*x_1*x_1,variance)</li>
     *      <li>N(w_10*x_0 + w_11*x_1 + w_12*x_1*x_0 + w_13*x_1*x_1*x_1,variance)</li>
     *      <li>N(w_20*x_0 + w_21*x_2 + w_22*x_1*x_0 + w_23*x_1*x_2*x_2,variance)</li>
     * </ul>
     * <p>
     * The features are drawn from a uniform distribution over the range.
     *
     * @param numSamples   The size of the output dataset.
     * @param yZeroWeights The feature weights for label y_0.
     * @param yOneWeights  The feature weights for label y_1.
     * @param yTwoWeights  The feature weights for label y_2.
     * @param threshold    The y threshold of each label.
     * @param negate       Should the computed value be negated before thresholding?
     * @param variance     The variance of the gaussian.
     * @param xMin         The minimum feature values (inclusive).
     * @param xMax         The maximum feature values (exclusive).
     * @param seed         The rng seed to use.
     * @return A dataset drawn from several gaussian generated labels.
     */
    public static MutableDataset<MultiLabel> generateDataset(int numSamples, float[] yZeroWeights, float[] yOneWeights, float[] yTwoWeights,
                                                      float[] threshold, boolean[] negate, float variance, float[] xMin, float[] xMax,
                                                      long seed) {
        MultiLabelGaussianDataSource source = new MultiLabelGaussianDataSource(numSamples, yZeroWeights, yOneWeights, yTwoWeights, threshold, negate, variance,
                xMin, xMax, seed);
        return new MutableDataset<>(source);
    }


    /**
     * Generates a multi label output drawn from a series of functions.
     * <p>
     * The functions are:
     * <ul>
     *     <li>y_0 is positive if N(w_00*x_0 + w_01*x_1 + w_02*x_1*x_0 + w_03*x_1*x_1*x_1,variance) &gt; threshold_0.</li>
     *     <li>y_1 is positive if N(w_10*x_0 + w_11*x_1 + w_12*x_1*x_0 + w_13*x_1*x_1*x_1,variance) &lt; threshold_1.</li>
     *     <li>y_2 is positive if N(w_20*x_0 + w_21*x_2 + w_22*x_1*x_0 + w_23*x_1*x_2*x_2,variance) &gt; threshold_2.</li>
     * </ul>
     * There are 4 features x_0, x_1, x_2, x_3. x_0 and x_1 are used by all three labels,
     * x_2 is only used by y_2, and x_3 is irrelevant.
     * <p>
     * By default y_1 is the inverse of y_0, and y_2 shares the same weights w_0 and w_2.
     * <ul>
     *     <li>y_0 weights = [1.0,1.0,1.0,1.0]</li>
     *     <li>y_1 weights = [1.0,1.0,1.0,1.0]</li>
     *     <li>y_2 weights = [1.0,-3.0,1.0,3.0]</li>
     *     <li>threshold = [0.0,0.0,2.0</li>
     * </ul>
     * <p>
     * The features are drawn from a uniform distribution over the range.
     *
     * @param numSamples The number of samples to draw.
     * @param seed       The RNG seed.
     * @return A dataset drawn from multiple Gaussians.
     */
    public static MultiLabelGaussianDataSource makeDefaultSource(int numSamples, long seed) {
        MultiLabelGaussianDataSource source = new MultiLabelGaussianDataSource();
        source.numSamples = numSamples;
        source.seed = seed;
        source.postConfig();
        return source;
    }

    /**
     * Provenance for {@link MultiLabelGaussianDataSource}.
     */
    public static class MultiLabelGaussianDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements ConfiguredDataSourceProvenance {
        private static final long serialVersionUID = 1L;

        /**
         * Constructs a provenance from the host data source.
         *
         * @param host The host to read.
         */
        MultiLabelGaussianDataSourceProvenance(MultiLabelGaussianDataSource host) {
            super(host, "DataSource");
        }

        /**
         * Constructs a provenance from the marshalled form.
         *
         * @param map The map of field values.
         */
        public MultiLabelGaussianDataSourceProvenance(Map<String, Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private MultiLabelGaussianDataSourceProvenance(ExtractedInfo info) {
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
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters, CLASS_NAME, StringProvenance.class, MultiLabelGaussianDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters, HOST_SHORT_NAME, StringProvenance.class, MultiLabelGaussianDataSourceProvenance.class.getSimpleName()).getValue();

            return new ExtractedInfo(className, hostTypeStringName, configuredParameters, Collections.emptyMap());
        }
    }
}
