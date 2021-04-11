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

package org.tribuo.interop.tensorflow;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.HashProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.exceptions.TensorFlowException;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Init;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.proto.framework.GraphDef;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.family.TNumber;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.SkeletalTrainerProvenance;
import org.tribuo.provenance.TrainerProvenance;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Trainer for Tensorflow. Expects the underlying Tensorflow graph to have named placeholders for
 * the inputs, ground truth outputs and a named output operation. The output operation should be
 * before any softmax or sigmoid non-linearities to allow the use of more optimized loss functions.
 * <p>
 * This trainer only works with graphs setup for minibatches. To recover single example training just use a batch size of 1.
 * <p>
 * This trainer uses the serialisation functionality in {@link TensorflowUtil}, as opposed to a SavedModel or a checkpoint.
 * <p>
 * N.B. Tensorflow support is experimental and may change without a major version bump.
 */
public final class TFTrainer<T extends Output<T>> implements Trainer<T> {

    private static final Logger logger = Logger.getLogger(TFTrainer.class.getName());

    @Config(mandatory=true,description="Path to the protobuf containing the graph.")
    private Path graphPath;

    private GraphDef graphDef;

    @Config(description="Test time batch size.")
    private int testBatchSize = 16;

    @Config(description="Name of the output operation before the loss.")
    private String outputName = TensorflowCheckpointTrainer.OUTPUT_NAME;

    @Config(description="Name of the init operation.")
    private String initName = Init.DEFAULT_NAME;

    @Config(mandatory=true,description="Feature extractor.")
    private ExampleTransformer<T> exampleTransformer;

    @Config(mandatory=true,description="Response extractor.")
    private OutputTransformer<T> outputTransformer;

    @Config(description="Minibatch size.")
    private int minibatchSize = 1;

    @Config(description="Number of SGD epochs to run.")
    private int epochs = 5;

    @Config(description="Logging interval to print out the loss.")
    private int loggingInterval = 100;

    @Config(mandatory=true,description="The gradient optimizer to use.")
    private GradientOptimiser optimizerEnum;

    @Config(mandatory=true,description="The gradient optimizer parameters.")
    private Map<String,Float> gradientParams;

    private int trainInvocationCounter = 0;

    /**
     * for olcut.
     */
    private TFTrainer() {}

    /**
     * Constructs a Trainer for a tensorflow graph.
     * @param graphPath The path to the graph protobuf. Must have the targets and placeholders specified above.
     * @param exampleTransformer The example transformer to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputTransformer The output transformer to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param minibatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @throws IOException If the graphPath is invalid or failed to load.
     */
    public TFTrainer(Path graphPath, ExampleTransformer<T> exampleTransformer, OutputTransformer<T> outputTransformer, int minibatchSize, int epochs, int testBatchSize) throws IOException {
        this.graphPath = graphPath;
        this.exampleTransformer = exampleTransformer;
        this.outputTransformer = outputTransformer;
        this.minibatchSize = minibatchSize;
        this.epochs = epochs;
        this.testBatchSize = testBatchSize;
        postConfig();
    }

    /**
     * Constructs a Trainer for a tensorflow graph.
     * @param graph The graph definition as a byte array. Must have the targets and placeholders specified above.
     * @param exampleTransformer The example transformer to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputTransformer The output transformer to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param minibatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     */
    public TFTrainer(Graph graph,
                     String outputName,
                     String initName,
                     GradientOptimiser optimizer,
                     Map<String,Float> gradientParams,
                     ExampleTransformer<T> exampleTransformer,
                     OutputTransformer<T> outputTransformer,
                     int minibatchSize,
                     int epochs,
                     int testBatchSize) {
        this.graphPath = null;
        this.graphDef = graph.toGraphDef();
        this.outputName = outputName;
        this.initName = initName;
        this.optimizerEnum = optimizer;
        this.gradientParams = Collections.unmodifiableMap(new HashMap<>(gradientParams));
        this.exampleTransformer = exampleTransformer;
        this.outputTransformer = outputTransformer;
        this.minibatchSize = minibatchSize;
        this.epochs = epochs;
        this.testBatchSize = testBatchSize;
        Shape outputShape = graph.operation(outputName).output(0).shape();
        if (outputShape.numDimensions() != 2) {
            throw new IllegalArgumentException("Expected a 2 dimensional output, found " + Arrays.toString(outputShape.asArray()));
        }
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() throws IOException {
        graphDef = GraphDef.parseFrom(new BufferedInputStream(new FileInputStream(graphPath.toFile())));
    }

    @Override
    public TFModel<T> train(Dataset<T> examples) {
        return train(examples,Collections.emptyMap());
    }

    @Override
    public TFModel<T> train(Dataset<T> examples, Map<String,Provenance> runProvenance) {
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<T> outputInfo = examples.getOutputIDInfo();
        ArrayList<Example<T>> batch = new ArrayList<>();
        trainInvocationCounter++;

        try (Graph graph = new Graph();
             Session session = new Session(graph)) {
            // Load in the graph definition
            graph.importGraphDef(graphDef);
            Ops tf = Ops.create(graph).withName("tribuo-internal");

            // Lookup output op
            Operand<TNumber> intermediateOutputOp = graph.operation(outputName).output(0);

            // Validate that the output op is the right shape
            Shape outputShape = intermediateOutputOp.shape();
            Shape expectedShape = Shape.of(minibatchSize,outputInfo.size());
            if (!outputShape.isCompatibleWith(expectedShape)) {
                throw new IllegalArgumentException("Incompatible output shape, expected " + expectedShape.toString() + " found " + outputShape.toString());
            }

            // Add target placeholder
            Placeholder<? extends TNumber> targetPlaceholder = tf.placeholder(TFloat32.class,Placeholder.shape(Shape.of(minibatchSize,outputInfo.size())));

            // Add loss, optimizer and output
            Op outputOp = outputTransformer.outputTransformFunction().apply(tf,intermediateOutputOp);
            Operand<TNumber> lossOp = outputTransformer.loss().apply(tf,new Pair<>(targetPlaceholder,intermediateOutputOp));
            Op optimizer = optimizerEnum.applyOptimizer(graph,lossOp,gradientParams);
            Init tribuoInit = tf.init();

            // Initialises the parameters.
            session.run(initName);

            // Initialise the gradient & output parameters.
            session.run(tribuoInit);
            logger.info("Initialised the model parameters");

            int interval = 0;
            for (int i = 0; i < epochs; i++) {
                logger.log(Level.INFO,"Starting epoch " + i);
                for (int j = 0; j < examples.size(); j += minibatchSize) {
                    batch.clear();
                    for (int k = j; k < (j+ minibatchSize) && k < examples.size(); k++) {
                        batch.add(examples.getExample(k));
                    }
                    //logger.info("Batch = " + batch.size());
                    ExampleTransformer.FeedDict input = exampleTransformer.transform(batch,featureMap);
                    Tensor target = outputTransformer.transform(batch,outputInfo);
                    Tensor lossTensor = input.feedInto(session.runner())
                            .feed(targetPlaceholder, target)
                            .addTarget(optimizer)
                            .fetch(lossOp)
                            .run().get(0);
                    if (interval % loggingInterval == 0) {
                        logger.log(Level.INFO, "Training loss = " + ((TFloat32) lossTensor).getFloat());
                    }
                    input.close();
                    target.close();
                    lossTensor.close();
                    interval++;
                }
            }

            //System.out.println("After training");
            //TensorflowModel.print(session);

            // This call **must** happen before the trainedGraphDef is generated.
            TensorflowUtil.annotateGraph(graph,session);

            GraphDef trainedGraphDef = graph.toGraphDef();

            Map<String, TensorflowUtil.TensorTuple> tensorMap = TensorflowUtil.serialise(graph,session);

            ModelProvenance modelProvenance = new ModelProvenance(TFModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), getProvenance(), runProvenance);
            TFModel<T> tfModel = new TFModel<>("tf-model", modelProvenance, featureMap,
                    outputInfo, trainedGraphDef, tensorMap, testBatchSize, initName, outputOp.op().name(), exampleTransformer, outputTransformer);
            return tfModel;
        } catch (TensorFlowException e) {
            logger.log(Level.SEVERE, "TensorFlow threw an error", e);
            throw new IllegalStateException(e);
        }
    }

    @Override
    public String toString() {
        String path = graphPath==null?"":graphPath.toString();
        return "TFTrainer(graphPath="+path+",exampleTransformer="
                +exampleTransformer.toString()+",outputTransformer="+outputTransformer.toString()
                +",minibatchSize="+ minibatchSize +",epochs="+ epochs +")";
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TFTrainerProvenance(this);
    }

    public static final class TFTrainerProvenance extends SkeletalTrainerProvenance {
        private static final long serialVersionUID = 1L;

        public static final String GRAPH_HASH = "graph-hash";
        public static final String GRAPH_LAST_MOD = "graph-last-modified";

        private final HashProvenance graphHash;
        private final DateTimeProvenance graphLastModified;

        <T extends Output<T>> TFTrainerProvenance(TFTrainer<T> host) {
            super(host);
            // instance parameters
            if (host.graphPath != null) {
                this.graphHash = new HashProvenance(DEFAULT_HASH_TYPE,GRAPH_HASH,ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE,host.graphPath));
                this.graphLastModified = new DateTimeProvenance(GRAPH_LAST_MOD, OffsetDateTime.ofInstant(Instant.ofEpochMilli(host.graphPath.toFile().lastModified()), ZoneId.systemDefault()));
            } else {
                this.graphHash = new HashProvenance(DEFAULT_HASH_TYPE,GRAPH_HASH,ProvenanceUtil.hashArray(DEFAULT_HASH_TYPE,host.graphDef.toByteArray()));
                this.graphLastModified = new DateTimeProvenance(GRAPH_LAST_MOD, OffsetDateTime.now());
            }
        }

        public TFTrainerProvenance(Map<String,Provenance> map) {
            this(extractTFProvenanceInfo(map));
        }

        private TFTrainerProvenance(ExtractedInfo info) {
            super(info);
            this.graphHash = (HashProvenance) info.instanceValues.get(GRAPH_HASH);
            this.graphLastModified = (DateTimeProvenance) info.instanceValues.get(GRAPH_LAST_MOD);
        }

        @Override
        public Map<String, PrimitiveProvenance<?>> getInstanceValues() {
            Map<String, PrimitiveProvenance<?>> map = super.getInstanceValues();

            map.put(graphHash.getKey(),graphHash);
            map.put(graphLastModified.getKey(),graphLastModified);

            return map;
        }

        protected static ExtractedInfo extractTFProvenanceInfo(Map<String,Provenance> map) {
            ExtractedInfo info = SkeletalTrainerProvenance.extractProvenanceInfo(map);
            info.instanceValues.put(GRAPH_HASH,ObjectProvenance.checkAndExtractProvenance(info.configuredParameters,GRAPH_HASH,HashProvenance.class, TFTrainerProvenance.class.getSimpleName()));
            info.instanceValues.put(GRAPH_LAST_MOD,ObjectProvenance.checkAndExtractProvenance(info.configuredParameters,GRAPH_LAST_MOD,DateTimeProvenance.class, TFTrainerProvenance.class.getSimpleName()));
            return info;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            if (!super.equals(o)) return false;
            TFTrainerProvenance pairs = (TFTrainerProvenance) o;
            return graphHash.equals(pairs.graphHash) && graphLastModified.equals(pairs.graphLastModified);
        }

        @Override
        public int hashCode() {
            return Objects.hash(super.hashCode(), graphHash, graphLastModified);
        }
    }
}
