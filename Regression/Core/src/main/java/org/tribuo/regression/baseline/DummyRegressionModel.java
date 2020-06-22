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

package org.tribuo.regression.baseline;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.baseline.DummyRegressionTrainer.DummyType;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

/**
 * A model which performs dummy regressions (e.g. constant output, gaussian sampled output, mean value, median, quartile).
 */
public class DummyRegressionModel extends Model<Regressor> {
    private static final long serialVersionUID = 2L;

    private final DummyType dummyType;

    private final Regressor output;

    private final long seed;

    private final Random rng;

    private final double[] means;

    private final double[] variances;

    private final String[] dimensionNames;

    DummyRegressionModel(ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo, long seed, double[] means, double[] variances, String[] names) {
        super("dummy-GAUSSIAN-regression", description, featureIDMap, outputIDInfo, false);
        this.dummyType = DummyType.GAUSSIAN;
        this.output = null;
        this.seed = seed;
        this.rng = new Random(seed);
        this.means = Arrays.copyOf(means,means.length);
        this.variances = Arrays.copyOf(variances,variances.length);
        this.dimensionNames = Arrays.copyOf(names,names.length);
    }

    DummyRegressionModel(ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo, DummyType dummyType, Regressor regressor) {
        super("dummy-"+dummyType+"-regression", description, featureIDMap, outputIDInfo, false);
        this.dummyType = dummyType;
        this.output = regressor;
        this.seed = Trainer.DEFAULT_SEED;
        this.rng = null;
        this.means = new double[0];
        this.variances = new double[0];
        this.dimensionNames = new String[0];
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        switch (dummyType) {
            case CONSTANT:
            case MEAN:
            case MEDIAN:
            case QUARTILE:
                return new Prediction<>(output,0,example);
            case GAUSSIAN: {
                Regressor.DimensionTuple[] dimensions = new Regressor.DimensionTuple[dimensionNames.length];
                for (int i = 0; i < dimensionNames.length; i++) {
                    double regressionValue = (rng.nextGaussian() * variances[i]) + means[i];
                    dimensions[i] = new Regressor.DimensionTuple(dimensionNames[i],regressionValue);
                }
                return new Prediction<>(new Regressor(dimensions), 0, example);
            }
            default:
                throw new IllegalStateException("Unknown dummyType " + dummyType);
        }
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        if (n != 0) {
            return Collections.singletonMap(Model.ALL_OUTPUTS, Collections.singletonList(new Pair<>(BIAS_FEATURE, 1.0)));
        } else {
            return Collections.emptyMap();
        }
    }

    @Override
    public Optional<Excuse<Regressor>> getExcuse(Example<Regressor> example) {
        return Optional.of(new Excuse<>(example,predict(example),getTopFeatures(1)));
    }

    @Override
    protected Model<Regressor> copy(String newName, ModelProvenance newProvenance) {
        switch (dummyType) {
            case GAUSSIAN:
                return new DummyRegressionModel(newProvenance,featureIDMap,outputIDInfo,seed,means,variances,dimensionNames);
            case CONSTANT:
            case MEAN:
            case MEDIAN:
            case QUARTILE:
                return new DummyRegressionModel(newProvenance,featureIDMap,outputIDInfo,dummyType,output.copy());
            default:
                throw new IllegalStateException("Unknown dummyType " + dummyType);
        }
    }
}
