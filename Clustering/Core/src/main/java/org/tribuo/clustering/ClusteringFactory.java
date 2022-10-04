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

package org.tribuo.clustering;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputFactory;
import org.tribuo.clustering.evaluation.ClusteringEvaluation;
import org.tribuo.clustering.evaluation.ClusteringEvaluator;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.protos.core.OutputFactoryProto;
import org.tribuo.provenance.OutputFactoryProvenance;

import java.util.HashMap;
import java.util.Map;

/**
 * A factory for making ClusterID related classes.
 * <p>
 * Parses the ClusterID by calling toString on the input then parsing it as an int.
 */
public final class ClusteringFactory implements OutputFactory<ClusterID> {
    private static final long serialVersionUID = 1L;

    /**
     * The sentinel unassigned cluster id, used when there is no ground truth clustering.
     */
    public static final ClusterID UNASSIGNED_CLUSTER_ID = new ClusterID(ClusterID.UNASSIGNED);

    private static final ClusteringFactoryProvenance provenance = new ClusteringFactoryProvenance();

    private static final ClusteringEvaluator evaluator = new ClusteringEvaluator();

    /**
     * ClusteringFactory is stateless and immutable, but we need to be able to construct them via the config system.
     */
    public ClusteringFactory() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static ClusteringFactory deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > 0) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + 0);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new ClusteringFactory();
    }

    @Override
    public OutputFactoryProto serialize() {
        return OutputFactoryProto.newBuilder().setVersion(0).setClassName(ClusteringFactory.class.getName()).build();
    }

    /**
     * Generates a ClusterID by calling toString on the input, then calling Integer.parseInt.
     * @param label An input value.
     * @param <V> The type of the input.
     * @return A ClusterID representing the data.
     */
    @Override
    public <V> ClusterID generateOutput(V label) {
        return new ClusterID(Integer.parseInt(label.toString()));
    }

    @Override
    public ClusterID getUnknownOutput() {
        return UNASSIGNED_CLUSTER_ID;
    }

    @Override
    public MutableOutputInfo<ClusterID> generateInfo() {
        return new MutableClusteringInfo();
    }

    /**
     * Unlike the other info types, clustering directly uses the integer IDs as the stored value,
     * so this mapping discards the cluster IDs and just uses the supplied integers.
     * @param mapping The mapping to use.
     * @return An {@link ImmutableOutputInfo} for the clustering.
     */
    @Override
    public ImmutableOutputInfo<ClusterID> constructInfoForExternalModel(Map<ClusterID,Integer> mapping) {
        // Validate inputs are dense
        OutputFactory.validateMapping(mapping);

        Map<Integer, MutableLong> countsMap = new HashMap<>();

        for (Map.Entry<ClusterID,Integer> e : mapping.entrySet()) {
            countsMap.put(e.getValue(),new MutableLong(1));
        }

        return new ImmutableClusteringInfo(countsMap);
    }

    @Override
    public Evaluator<ClusterID, ClusteringEvaluation> getEvaluator() {
        return evaluator;
    }

    @Override
    public Class<ClusterID> getTypeWitness() {
        return ClusterID.class;
    }

    @Override
    public int hashCode() {
        return "ClusteringFactory".hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof ClusteringFactory;
    }

    @Override
    public OutputFactoryProvenance getProvenance() {
        return provenance;
    }

    /**
     * Provenance for {@link ClusteringFactory}.
     */
    public final static class ClusteringFactoryProvenance implements OutputFactoryProvenance {
        private static final long serialVersionUID = 1L;

        /**
         * Creates a clustering factory provenance.
         */
        ClusteringFactoryProvenance() {}

        /**
         * Rebuilds a clustering factory provenance from the marshalled form.
         * @param map The map (which should be empty).
         */
        public ClusteringFactoryProvenance(Map<String, Provenance> map) { }

        @Override
        public String getClassName() {
            return ClusteringFactory.class.getName();
        }

        @Override
        public String toString() {
            return generateString("OutputFactory");
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof ClusteringFactoryProvenance;
        }

        @Override
        public int hashCode() {
            return 31;
        }
    }
}
