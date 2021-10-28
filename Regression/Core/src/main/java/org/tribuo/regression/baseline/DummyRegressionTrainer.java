/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DoubleProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.EnumProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.LongProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.regression.Regressor;
import org.tribuo.util.Util;

import java.time.OffsetDateTime;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * A trainer for simple baseline regressors. Use this only for comparison purposes, if you can't beat these
 * baselines, your ML system doesn't work.
 */
public final class DummyRegressionTrainer implements Trainer<Regressor> {

    /**
     * Types of dummy regression model.
     */
    public enum DummyType {
        /**
         * Returns the mean of the training data outputs.
         */
        MEAN,
        /**
         * Returns the median of the training data outputs.
         */
        MEDIAN,
        /**
         * Returns the training data output at the specified fraction of the sorted output.
         */
        QUARTILE,
        /**
         * Returns the specified constant value.
         */
        CONSTANT,
        /**
         * Samples from a Gaussian using the means and variances from the training data.
         */
        GAUSSIAN
    }

    @Config(mandatory = true, description="Type of dummy regressor.")
    private DummyType dummyType;

    @Config(description="Constant value to use for the constant regressor.")
    private double constantValue = Double.NaN;

    @Config(description="Quartile to use.")
    private double quartile = Double.NaN;

    @Config(description="The seed for the RNG.")
    private long seed = 1L;

    private int trainInvocationCounter = 0;

    private DummyRegressionTrainer() { }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if ((dummyType == DummyType.CONSTANT) && (Double.isNaN(constantValue))) {
            throw new PropertyException("","constantValue","Please supply a constant value when using the type CONSTANT.");
        }
        if ((dummyType == DummyType.QUARTILE) && ((quartile < 0.) || (quartile > 1.0))) {
            throw new PropertyException("","quartile","Please supply a quartile between zero and one when using the type QUARTILE.");
        }
    }

    @Override
    public DummyRegressionModel train(Dataset<Regressor> examples, Map<String, Provenance> instanceProvenance) {
        return train(examples, instanceProvenance, INCREMENT_INVOCATION_COUNT);
    }

    @Override
    public DummyRegressionModel train(Dataset<Regressor> examples, Map<String, Provenance> instanceProvenance, int invocationCount) {
        if(invocationCount != INCREMENT_INVOCATION_COUNT) {
            setInvocationCount(invocationCount);
        }
        ModelProvenance provenance = new ModelProvenance(DummyRegressionModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), getProvenance(), instanceProvenance);
        trainInvocationCounter++;
        ImmutableOutputInfo<Regressor> outputInfo = examples.getOutputIDInfo();
        Set<Regressor> domain = outputInfo.getDomain();
        double[][] outputs = new double[outputInfo.size()][examples.size()];
        int i = 0;
        for (Example<Regressor> e : examples) {
            for (Regressor.DimensionTuple r : e.getOutput()) {
                int id = outputInfo.getID(r);
                outputs[id][i] = r.getValue();
            }
            i++;
        }
        Regressor regressor;
        switch (dummyType) {
            case CONSTANT: {
                Regressor.DimensionTuple[] output = new Regressor.DimensionTuple[outputs.length];
                for (Regressor r : domain) {
                    int id = outputInfo.getID(r);
                    output[id] = new Regressor.DimensionTuple(r.getNames()[0],constantValue);
                }
                regressor = new Regressor(output);
                return new DummyRegressionModel(provenance,examples.getFeatureIDMap(),outputInfo,dummyType,regressor);
            }
            case MEAN: {
                Regressor.DimensionTuple[] output = new Regressor.DimensionTuple[outputs.length];
                for (Regressor r : domain) {
                    int id = outputInfo.getID(r);
                    output[id] = new Regressor.DimensionTuple(r.getNames()[0],Util.mean(outputs[id]));
                }
                regressor = new Regressor(output);
                return new DummyRegressionModel(provenance,examples.getFeatureIDMap(),outputInfo,dummyType,regressor);
            }
            case MEDIAN: {
                Regressor.DimensionTuple[] output = new Regressor.DimensionTuple[outputs.length];
                for (Regressor r : domain) {
                    int id = outputInfo.getID(r);
                    Arrays.sort(outputs[id]);
                    output[id] = new Regressor.DimensionTuple(r.getNames()[0],outputs[id][outputs[id].length/2]);
                }
                regressor = new Regressor(output);
                return new DummyRegressionModel(provenance,examples.getFeatureIDMap(),outputInfo,dummyType,regressor);
            }
            case QUARTILE: {
                Regressor.DimensionTuple[] output = new Regressor.DimensionTuple[outputs.length];
                for (Regressor r : domain) {
                    int id = outputInfo.getID(r);
                    Arrays.sort(outputs[id]);
                    output[id] = new Regressor.DimensionTuple(r.getNames()[0],outputs[id][(int) (quartile*outputs[id].length)]);
                }
                regressor = new Regressor(output);
                return new DummyRegressionModel(provenance,examples.getFeatureIDMap(),outputInfo,dummyType,regressor);
            }
            case GAUSSIAN: {
                double[] means = new double[outputs.length];
                double[] variances = new double[outputs.length];
                String[] names = new String[outputs.length];
                for (Regressor r : domain) {
                    int id = outputInfo.getID(r);
                    names[id] = r.getNames()[0];
                    Pair<Double,Double> meanVariance = Util.meanAndVariance(outputs[id]);
                    means[id] = meanVariance.getA();
                    variances[id] = meanVariance.getB();
                }
                return new DummyRegressionModel(provenance,examples.getFeatureIDMap(),outputInfo,seed,means,variances,names);
            }
            default:
                throw new IllegalStateException("Unknown dummyType " + dummyType);
        }
    }

    @Override
    public String toString() {
        switch (dummyType) {
            case CONSTANT:
                return "DummyRegressionTrainer(dummyType=CONSTANT,constantValue="+constantValue+")";
            case MEAN:
                return "DummyRegressionTrainer(dummyType=MEAN)";
            case MEDIAN:
                return "DummyRegressionTrainer(dummyType=MEDIAN)";
            case QUARTILE:
                return "DummyRegressionTrainer(dummyType=QUARTILE,quartile="+quartile+")";
            case GAUSSIAN:
                return "DummyRegressionTrainer(dummyType=GAUSSIAN,seed="+seed+")";
            default:
                return "DummyRegressionTrainer(dummyType="+dummyType+")";
        }
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public synchronized void setInvocationCount(int invocationCount){
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }
        this.trainInvocationCounter = invocationCount;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }

    /**
     * Creates a trainer which create models which return a fixed value.
     * @param value The value to return
     * @return A regression trainer.
     */
    public static DummyRegressionTrainer createConstantTrainer(double value) {
        DummyRegressionTrainer trainer = new DummyRegressionTrainer();
        trainer.dummyType = DummyType.CONSTANT;
        trainer.constantValue = value;
        return trainer;
    }

    /**
     * Creates a trainer which create models which sample the output from a gaussian distribution fit to the training data.
     * @param seed The RNG seed.
     * @return A regression trainer.
     */
    public static DummyRegressionTrainer createGaussianTrainer(long seed) {
        DummyRegressionTrainer trainer = new DummyRegressionTrainer();
        trainer.dummyType = DummyType.GAUSSIAN;
        trainer.seed = seed;
        return trainer;
    }

    /**
     * Creates a trainer which create models which return the mean of the training data.
     * @return A regression trainer.
     */
    public static DummyRegressionTrainer createMeanTrainer() {
        DummyRegressionTrainer trainer = new DummyRegressionTrainer();
        trainer.dummyType = DummyType.MEAN;
        return trainer;
    }

    /**
     * Creates a trainer which create models which return the median of the training data.
     * @return A regression trainer.
     */
    public static DummyRegressionTrainer createMedianTrainer() {
        DummyRegressionTrainer trainer = new DummyRegressionTrainer();
        trainer.dummyType = DummyType.MEDIAN;
        return trainer;
    }

    /**
     * Creates a trainer which create models which return the value at the specified fraction of the sorted training data.
     * @param value The quartile value.
     * @return A regression trainer.
     */
    public static DummyRegressionTrainer createQuartileTrainer(double value) {
        if (Double.isNaN(value) || value < 0.0 || value > 1.0) {
            throw new IllegalArgumentException("Please provide an appropriate value between 0.0 and 1.0, found " + value);
        }
        DummyRegressionTrainer trainer = new DummyRegressionTrainer();
        trainer.dummyType = DummyType.QUARTILE;
        trainer.quartile = value;
        return trainer;
    }

    /**
     * Provenance for {@link DummyRegressionTrainer}.
     */
    @Deprecated
    public final static class DummyRegressionTrainerProvenance implements TrainerProvenance {
        private static final long serialVersionUID = 1L;

        private final String className;
        private final DummyType dummyType;
        private final long seed;
        private final double constantValue;
        private final double quartile;

        /**
         * Constructs a provenance from the host.
         * @param host The host trainer.
         */
        public DummyRegressionTrainerProvenance(DummyRegressionTrainer host) {
            this.className = host.getClass().getName();
            this.dummyType = host.dummyType;
            this.seed = host.seed;
            this.constantValue = host.constantValue;
            this.quartile = host.quartile;
        }

        /**
         * Constructs a provenance from the marshalled form.
         * @param map The map of field values.
         */
        public DummyRegressionTrainerProvenance(Map<String, Provenance> map) {
            className = ObjectProvenance.checkAndExtractProvenance(map,CLASS_NAME, StringProvenance.class, DummyRegressionTrainerProvenance.class.getSimpleName()).getValue();
            dummyType = (DummyType) ObjectProvenance.checkAndExtractProvenance(map,"dummyType", EnumProvenance.class, DummyRegressionTrainerProvenance.class.getSimpleName()).getValue();
            seed = ObjectProvenance.checkAndExtractProvenance(map,"seed", LongProvenance.class, DummyRegressionTrainerProvenance.class.getSimpleName()).getValue();
            constantValue = ObjectProvenance.checkAndExtractProvenance(map,"constantValue", DoubleProvenance.class, DummyRegressionTrainerProvenance.class.getSimpleName()).getValue();
            quartile = ObjectProvenance.checkAndExtractProvenance(map,"quartile", DoubleProvenance.class, DummyRegressionTrainerProvenance.class.getSimpleName()).getValue();
        }

        @Override
        public Map<String, Provenance> getConfiguredParameters() {
            Map<String, Provenance> map = new HashMap<>();

            map.put("dummyType",new EnumProvenance<>("dummyType",dummyType));
            map.put("constantValue",new DoubleProvenance("constantValue",constantValue));
            map.put("quartile",new DoubleProvenance("quartile",quartile));
            map.put("seed",new LongProvenance("seed",seed));

            return map;
        }

        @Override
        public String getClassName() {
            return className;
        }

        @Override
        public String toString() {
            return generateString("Trainer");
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            DummyRegressionTrainerProvenance pairs = (DummyRegressionTrainerProvenance) o;
            return seed == pairs.seed &&
                    Double.compare(pairs.constantValue, constantValue) == 0 &&
                    Double.compare(pairs.quartile, quartile) == 0 &&
                    className.equals(pairs.className) &&
                    dummyType == pairs.dummyType;
        }

        @Override
        public int hashCode() {
            return Objects.hash(className, dummyType, seed, constantValue, quartile);
        }
    }
}
