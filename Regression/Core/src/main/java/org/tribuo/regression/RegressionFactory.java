/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.CharProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputFactory;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.provenance.OutputFactoryProvenance;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.protos.RegressionFactoryProto;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A factory for creating {@link Regressor}s and {@link RegressionInfo}s.
 * <p>
 * It parses the regression dimensions by toStringing the input and calling {@link Regressor#parseString}, unless
 * the input is a collection, in which case it extracts the elements.
 * <p>
 * This OutputFactory has mutable state, namely the character which the dimension input is split on.
 * In most cases the default {@link RegressionFactory#DEFAULT_SPLIT_CHAR} is fine.
 */
public final class RegressionFactory implements OutputFactory<Regressor> {
    private static final long serialVersionUID = 2L;

    /**
     * The default character to split the string form of a multidimensional regressor.
     */
    public static final char DEFAULT_SPLIT_CHAR = ',';

    @Config(description="The character to split the dimensions on.")
    private char splitChar = DEFAULT_SPLIT_CHAR;

    /**
     * The sentinel unknown regressor, used when there is no ground truth regressor value.
     */
    public static final Regressor UNKNOWN_REGRESSOR = new Regressor(new String[]{"UNKNOWN"}, new double[]{Double.NaN});

    /**
     * @deprecated Deprecated when regression was made multidimensional by default.
     * Use {@link #UNKNOWN_REGRESSOR} instead.
     */
    @Deprecated
    public static final Regressor UNKNOWN_MULTIPLE_REGRESSOR = UNKNOWN_REGRESSOR;

    private RegressionFactoryProvenance provenance;

    private static final RegressionEvaluator evaluator = new RegressionEvaluator();

    /**
     * Builds a regression factory using the default split character {@link RegressionFactory#DEFAULT_SPLIT_CHAR}.
     */
    public RegressionFactory() {
        this.provenance = new RegressionFactoryProvenance(splitChar);
    }

    /**
     * Sets the split character used to parse {@link Regressor} instances from Strings.
     * @param splitChar The split character.
     */
    public RegressionFactory(char splitChar) {
        this.splitChar = splitChar;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        this.provenance = new RegressionFactoryProvenance(splitChar);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static RegressionFactory deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        RegressionFactoryProto proto = message.unpack(RegressionFactoryProto.class);
        String split = proto.getSplitChar();
        if (split.length() != 1) {
            throw new IllegalArgumentException("Invalid protobuf, splitChar must be a single character, found '" + split + "'");
        }

        return new RegressionFactory(split.charAt(0));
    }

    @Override
    public OutputFactoryProto serialize() {
        OutputFactoryProto.Builder outputBuilder = OutputFactoryProto.newBuilder();

        outputBuilder.setClassName(RegressionFactory.class.getName());
        outputBuilder.setVersion(0);

        RegressionFactoryProto.Builder data = RegressionFactoryProto.newBuilder();
        data.setSplitChar(""+splitChar);

        outputBuilder.setSerializedData(Any.pack(data.build()));

        return outputBuilder.build();
    }

    /**
     * Parses the Regressor value either by toStringing the input and calling {@link Regressor#parseString}
     * or if it's a collection iterating over the elements calling toString on each element in turn and using
     * {@link Regressor#parseElement}.
     * @param label An input value.
     * @param <V> The type of the input value.
     * @return A MultipleRegressor with sentinel variances.
     */
    @Override
    public <V> Regressor generateOutput(V label) {
        if (label instanceof Collection) {
            Collection<?> c = (Collection<?>) label;
            List<Pair<String,Double>> dimensions = new ArrayList<>();
            int i = 0;
            for (Object o : c) {
                dimensions.add(Regressor.parseElement(i,o.toString()));
                i++;
            }
            return Regressor.createFromPairList(dimensions);
        } else {
            return Regressor.parseString(label.toString(), splitChar);
        }
    }

    @Override
    public Regressor getUnknownOutput() {
        return UNKNOWN_REGRESSOR;
    }

    @Override
    public MutableOutputInfo<Regressor> generateInfo() {
        return new MutableRegressionInfo();
    }

    @Override
    public ImmutableOutputInfo<Regressor> constructInfoForExternalModel(Map<Regressor,Integer> mapping) {
        // Validate inputs are dense
        OutputFactory.validateMapping(mapping);

        // Coalesce all the mappings into a single Regressor.
        String[] names = new String[mapping.size()];
        double[] values = new double[mapping.size()];
        double[] variances = new double[mapping.size()];
        int i = 0;
        for (Map.Entry<Regressor,Integer> m : mapping.entrySet()) {
            Regressor r = m.getKey();
            if (r.size() != 1) {
                throw new IllegalArgumentException("Expected to find a DimensionTuple, found multiple dimensions for a single integer. Found = " + r);
            }
            names[i] = r.getNames()[0];
            values[i] = r.getValues()[0];
            variances[i] = r.getVariances()[0];
            i++;
        }

        Regressor newRegressor = new Regressor(names,values,variances);

        MutableRegressionInfo info = new MutableRegressionInfo();

        info.observe(newRegressor);

        return new ImmutableRegressionInfo(info,mapping);
    }

    @Override
    public Evaluator<Regressor, RegressionEvaluation> getEvaluator() {
        return evaluator;
    }

    @Override
    public Class<Regressor> getTypeWitness() {
        return Regressor.class;
    }

    @Override
    public int hashCode() {
        return "RegressionFactory".hashCode() ^ Character.hashCode(splitChar);
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof RegressionFactory && splitChar == ((RegressionFactory) obj).splitChar;
    }

    @Override
    public OutputFactoryProvenance getProvenance() {
        return provenance;
    }

    /**
     * Provenance for {@link RegressionFactory}.
     */
    public final static class RegressionFactoryProvenance implements OutputFactoryProvenance {
        private static final long serialVersionUID = 1L;

        private final char splitChar;

        /**
         * Constructs a provenance for the factory, reading it's split character.
         * @param splitChar The split character used by the factory.
         */
        RegressionFactoryProvenance(char splitChar) {
            this.splitChar = splitChar;
        }

        /**
         * Constructs a provenance from it's marshalled form.
         * @param map The provenance map, containing a splitChar field.
         */
        public RegressionFactoryProvenance(Map<String,Provenance> map) {
            this.splitChar = ((CharProvenance)map.get("splitChar")).getValue();
        }

        @Override
        public Map<String, Provenance> getConfiguredParameters() {
            return Collections.singletonMap("splitChar",new CharProvenance("splitChar",splitChar));
        }

        @Override
        public String getClassName() {
            return RegressionFactory.class.getName();
        }

        @Override
        public String toString() {
            return generateString("OutputFactory");
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof RegressionFactoryProvenance)) return false;
            RegressionFactoryProvenance pairs = (RegressionFactoryProvenance) o;
            return splitChar == pairs.splitChar;
        }

        @Override
        public int hashCode() {
            return Objects.hash(splitChar);
        }
    }
}
