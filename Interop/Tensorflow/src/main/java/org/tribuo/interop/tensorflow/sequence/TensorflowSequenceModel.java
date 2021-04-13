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

package org.tribuo.interop.tensorflow.sequence;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tensorflow.proto.framework.GraphDef;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.interop.tensorflow.TensorMap;
import org.tribuo.interop.tensorflow.TensorFlowUtil;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.sequence.SequenceExample;
import org.tribuo.sequence.SequenceModel;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * A Tensorflow model which implements SequenceModel, suitable for use in sequential prediction tasks.
 */
public class TensorflowSequenceModel<T extends Output<T>> extends SequenceModel<T> implements AutoCloseable {

    private static final long serialVersionUID = 200L;

    private transient Graph modelGraph = null;
    private transient Session session = null;

    protected final SequenceExampleTransformer<T> exampleTransformer;
    protected final SequenceOutputTransformer<T> outputTransformer;

    protected final String initOp;
    protected final String predictOp;

    TensorflowSequenceModel(String name,
                            ModelProvenance description,
                            ImmutableFeatureMap featureIDMap,
                            ImmutableOutputInfo<T> outputIDMap,
                            GraphDef graphDef,
                            SequenceExampleTransformer<T> exampleTransformer,
                            SequenceOutputTransformer<T> outputTransformer,
                            String initOp,
                            String predictOp,
                            Map<String, TensorFlowUtil.TensorTuple> tensorMap
    ) {
        super(name, description, featureIDMap, outputIDMap);
        this.exampleTransformer = exampleTransformer;
        this.outputTransformer = outputTransformer;
        this.initOp = initOp;
        this.predictOp = predictOp;
        this.modelGraph = new Graph();
        this.modelGraph.importGraphDef(graphDef);
        this.session = new Session(modelGraph);

        // Initialises the parameters.
        session.runner().addTarget(initOp).run();
        TensorFlowUtil.deserialise(session, tensorMap);
    }

    @Override
    public List<Prediction<T>> predict(SequenceExample<T> example) {
        TensorMap feed = exampleTransformer.encode(example, featureIDMap);
        Session.Runner runner = session.runner();
        runner = feed.feedInto(runner);
        try (Tensor outputTensor = runner
                .fetch(predictOp)
                .run()
                .get(0)) {
            List<Prediction<T>> prediction = outputTransformer.decode(outputTensor, example, outputIDMap);
            //
            // Close the input tensors
            feed.close();
            return prediction;
        }
    }

    /**
     * Returns an empty map, as the top features are not well defined for most Tensorflow models.
     */
    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int i) {
        return Collections.emptyMap();
    }

    /**
     * Close the session and graph if they exist.
     */
    @Override
    public void close() {
        if (session != null) {
            session.close();
        }
        if (modelGraph != null) {
            modelGraph.close();
        }
    }

    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
        byte[] modelBytes = modelGraph.toGraphDef().toByteArray();
        out.writeObject(modelBytes);
        Map<String, TensorFlowUtil.TensorTuple> tensorMap = TensorFlowUtil.serialise(modelGraph, session);
        out.writeObject(tensorMap);
    }

    @SuppressWarnings("unchecked") //deserialising a typed map
    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        byte[] modelBytes = (byte[]) in.readObject();
        Map<String, TensorFlowUtil.TensorTuple> tensorMap = (Map<String, TensorFlowUtil.TensorTuple>) in.readObject();
        modelGraph = new Graph();
        modelGraph.importGraphDef(GraphDef.parseFrom(modelBytes));
        session = new Session(modelGraph);
        // Initialises the parameters.
        session.runner().addTarget(initOp).run();
        TensorFlowUtil.deserialise(session,tensorMap);
    }
}