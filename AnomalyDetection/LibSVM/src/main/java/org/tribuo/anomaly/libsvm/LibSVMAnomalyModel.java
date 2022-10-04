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

package org.tribuo.anomaly.libsvm;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.anomaly.Event;
import org.tribuo.anomaly.libsvm.protos.LibSVMAnomalyModelProto;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.LibSVMTrainer;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;

import java.util.Collections;
import java.util.List;

/**
 * An anomaly detection model that uses an underlying libSVM model to make the
 * predictions.
 * <p>
 * See:
 * <pre>
 * Chang CC, Lin CJ.
 * "LIBSVM: a library for Support Vector Machines"
 * ACM transactions on intelligent systems and technology (TIST), 2011.
 * </pre>
 * <p>
 * and for the anomaly detection algorithm:
 * <pre>
 * Schölkopf B, Platt J, Shawe-Taylor J, Smola A J, Williamson R C.
 * "Estimating the support of a high-dimensional distribution"
 * Neural Computation, 2001, 1443-1471.
 * </pre>
 */
public class LibSVMAnomalyModel extends LibSVMModel<Event> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    LibSVMAnomalyModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Event> labelIDMap, List<svm_model> models) {
        super(name, description, featureIDMap, labelIDMap, models.get(0).param.probability == 1, models);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static LibSVMAnomalyModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        LibSVMAnomalyModelProto proto = message.unpack(LibSVMAnomalyModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Event.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not an anomaly detection domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Event> outputDomain = (ImmutableOutputInfo<Event>) carrier.outputDomain();

        svm_model model = deserializeModel(proto.getModel());

        return new LibSVMAnomalyModel(carrier.name(),carrier.provenance(),carrier.featureDomain(),outputDomain,Collections.singletonList(model));
    }

    /**
     * Returns the number of support vectors.
     * @return The number of support vectors.
     */
    public int getNumberOfSupportVectors() {
        return models.get(0).SV.length;
    }

    @Override
    public Prediction<Event> predict(Example<Event> example) {
        svm_node[] features = LibSVMTrainer.exampleToNodes(example, featureIDMap, null);
        // Bias feature is always set by the library.
        if (features.length == 0) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }
        double[] score = new double[1];
        double prediction = svm.svm_predict_values(models.get(0), features, score);
        if (prediction < 0.0) {
            return new Prediction<>(new Event(Event.EventType.ANOMALOUS,score[0]),features.length,example);
        } else {
            return new Prediction<>(new Event(Event.EventType.EXPECTED,score[0]),features.length,example);
        }
    }

    @Override
    protected LibSVMAnomalyModel copy(String newName, ModelProvenance newProvenance) {
        return new LibSVMAnomalyModel(newName,newProvenance,featureIDMap,outputIDInfo, Collections.singletonList(LibSVMModel.copyModel(models.get(0))));
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<Event> carrier = createDataCarrier();

        LibSVMAnomalyModelProto.Builder modelBuilder = LibSVMAnomalyModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setModel(serializeModel(models.get(0)));

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(LibSVMAnomalyModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }
}
