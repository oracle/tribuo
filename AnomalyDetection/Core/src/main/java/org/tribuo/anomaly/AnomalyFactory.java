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

package org.tribuo.anomaly;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputFactory;
import org.tribuo.anomaly.Event.EventType;
import org.tribuo.anomaly.evaluation.AnomalyEvaluation;
import org.tribuo.anomaly.evaluation.AnomalyEvaluator;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.provenance.OutputFactoryProvenance;

import java.util.Map;

/**
 * A factory for generating events.
 */
public final class AnomalyFactory implements OutputFactory<Event> {
    private static final long serialVersionUID = 1L;

    /**
     * The unknown event. Used at inference time.
     */
    public static final Event UNKNOWN_EVENT = new Event(EventType.UNKNOWN);

    /**
     * The expected event. Used for things which are not anomalous.
     */
    public static final Event EXPECTED_EVENT = new Event(EventType.EXPECTED);

    /**
     * The anomalous event. It's anomalous.
     */
    public static final Event ANOMALOUS_EVENT = new Event(EventType.ANOMALOUS);

    private static final AnomalyEvaluator evaluator = new AnomalyEvaluator();

    /**
     * Create an AnomalyFactory.
     */
    public AnomalyFactory() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static AnomalyFactory deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new AnomalyFactory();
    }

    @Override
    public OutputFactoryProto serialize() {
        return OutputFactoryProto.newBuilder().setVersion(0).setClassName(AnomalyFactory.class.getName()).build();
    }

    @Override
    public <V> Event generateOutput(V label) {
        if (label.toString().equalsIgnoreCase(EventType.ANOMALOUS.toString())) {
            return ANOMALOUS_EVENT;
        } else {
            return EXPECTED_EVENT;
        }
    }

    @Override
    public Event getUnknownOutput() {
        return UNKNOWN_EVENT;
    }

    @Override
    public MutableOutputInfo<Event> generateInfo() {
        return new MutableAnomalyInfo();
    }

    @Override
    public ImmutableOutputInfo<Event> constructInfoForExternalModel(Map<Event,Integer> mapping) {
        // Validate inputs are dense
        OutputFactory.validateMapping(mapping);

        Integer expectedMapping = mapping.get(EXPECTED_EVENT);
        Integer anomalousMapping = mapping.get(ANOMALOUS_EVENT);

        if (((expectedMapping != null) && (expectedMapping != EventType.EXPECTED.getID())) ||
        ((anomalousMapping != null) && anomalousMapping != EventType.ANOMALOUS.getID())){
            throw new IllegalArgumentException("Anomaly detection requires that anomalous events have id " + EventType.ANOMALOUS.getID() + ", and expected events have id " + EventType.EXPECTED.getID());
        }

        MutableAnomalyInfo info = new MutableAnomalyInfo();
        return info.generateImmutableOutputInfo();
    }

    @Override
    public Evaluator<Event, AnomalyEvaluation> getEvaluator() {
        return evaluator;
    }

    @Override
    public Class<Event> getTypeWitness() {
        return Event.class;
    }

    @Override
    public int hashCode() {
        return "AnomalyFactory".hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof AnomalyFactory;
    }

    /**
     * Generate provenance for this anomaly factory.
     * @return The provenance.
     */
    @Override
    public OutputFactoryProvenance getProvenance() {
        return new AnomalyFactoryProvenance();
    }

    /**
     * Provenance for {@link AnomalyFactory}.
     */
    public final static class AnomalyFactoryProvenance implements OutputFactoryProvenance {
        private static final long serialVersionUID = 1L;

        /**
         * Constructs an anomaly factory provenance.
         */
        AnomalyFactoryProvenance() {}

        /**
         * Constructs an anomaly factory provenance from the marshalled form.
         * @param map An empty map.
         */
        public AnomalyFactoryProvenance(Map<String, Provenance> map) { }

        @Override
        public String getClassName() {
            return AnomalyFactory.class.getName();
        }

        @Override
        public String toString() {
            return generateString("OutputFactory");
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof AnomalyFactoryProvenance;
        }

        @Override
        public int hashCode() {
            return 31;
        }
    }
}
