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

package org.tribuo.transform.transformations;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;

import org.tribuo.protos.core.IDFTransformerProto;
import org.tribuo.protos.core.TransformerProto;
import org.tribuo.transform.TransformStatistics;
import org.tribuo.transform.Transformation;
import org.tribuo.transform.TransformationProvenance;
import org.tribuo.transform.Transformer;

/**
 * A feature transformation that computes the IDF for features and then transforms
 * them with a TF-IDF weighting.
 */
public class IDFTransformation implements Transformation {
    
    private TransformationProvenance provenance;

    /**
     * Constructs an IDFTransformation.
     */
    public IDFTransformation() {}

    @Override
    public TransformStatistics createStats() {
        return new IDFStatistics();
    }

    @Override
    public TransformationProvenance getProvenance() {
        if (provenance == null) {
            provenance = new IDFTransformationProvenance();
        }
        return provenance;
    }
    
    private static class IDFStatistics implements TransformStatistics {
        
        /**
         * The document frequency for the feature that this statistic is
         * associated with. This is a count of the number of examples that the
         * feature occurs in.
         */
        private int df;
        
        /**
         * The number of examples that the feature did not occur in.
         */
        private int sparseObservances;
        

        @Override
        public void observeValue(double value) {
            //
            // One more document (i.e., an example) has this feature.
            df++;
        }

        @Override
        @Deprecated
        public void observeSparse() {
            sparseObservances++;
        }

        @Override
        public void observeSparse(int count) {
            sparseObservances = count;
        }

        @Override
        public Transformer generateTransformer() {
            return new IDFTransformer(df, df+sparseObservances);
        }
        
    }
    
    static class IDFTransformer implements Transformer {
        private static final long serialVersionUID = 1L;

        private final double df;
        
        private final double N;

        /**
         * Constructs an IDFTransformer using the supplied parameters.
         * @param df The document frequency.
         * @param N The number of documents.
         */
        IDFTransformer(int df, int N) {
            if ((df < 0) || (N < 0)) {
                throw new IllegalArgumentException("Both df and N must be positive");
            }
            this.df = df;
            this.N = N;
        }

        /**
         * Deserialization factory.
         * @param version The serialized object version.
         * @param className The class name.
         * @param message The serialized data.
         * @throws InvalidProtocolBufferException If the message is not a {@link IDFTransformerProto}.
         */
        static IDFTransformer deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
            IDFTransformerProto proto = message.unpack(IDFTransformerProto.class);
            if (version == 0) {
                return new IDFTransformer((int)proto.getDf(), (int)proto.getN());
            } else {
                throw new IllegalArgumentException("Unknown version " + version + " expected {0}");
            }
        }

        @Override
        public double transform(double tf) {
            return Math.log(N / df) * (1 + Math.log(tf));
        }

        @Override
        public TransformerProto serialize() {
            TransformerProto.Builder protoBuilder = TransformerProto.newBuilder();

            protoBuilder.setVersion(0);
            protoBuilder.setClassName(this.getClass().getName());

            IDFTransformerProto transformProto = IDFTransformerProto.newBuilder()
                    .setDf(df).setN(N).build();
            protoBuilder.setSerializedData(Any.pack(transformProto));

            return protoBuilder.build();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            IDFTransformer that = (IDFTransformer) o;
            return Double.compare(that.df, df) == 0 && Double.compare(that.N, N) == 0;
        }

        @Override
        public int hashCode() {
            return Objects.hash(df, N);
        }
    }

    /**
     * Provenance for {@link IDFTransformation}.
     */
    public final static class IDFTransformationProvenance implements TransformationProvenance {
        private static final long serialVersionUID = 1L;

        IDFTransformationProvenance() { }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        // IDFTransformation has no state to record.
        public IDFTransformationProvenance(Map<String,Provenance> map) { }

        @Override
        public Map<String, Provenance> getConfiguredParameters() {
            return Collections.emptyMap();
        }

        @Override
        public String getClassName() {
            return IDFTransformation.class.getName();
        }
        
    }

}
