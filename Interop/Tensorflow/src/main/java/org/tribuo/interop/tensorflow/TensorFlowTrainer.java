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
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.HashProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
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
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
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
 * This trainer uses the serialisation functionality in {@link TensorFlowUtil}, as opposed to a SavedModel or a checkpoint.
 * <p>
 * N.B. Tensorflow support is experimental and may change without a major version bump.
 */
public final class TensorFlowTrainer<T extends Output<T>> implements Trainer<T> {

    private static final Logger logger = Logger.getLogger(TensorFlowTrainer.class.getName());

    /**
     * The model format to emit.
     */
    public enum TFModelFormat {
        /**
         * Saves the model inside the Tribuo object, emits a {@link TensorFlowNativeModel}.
         */
        TRIBUO_NATIVE,
        /**
         * Saves the model state inside a TensorFlow checkpoint, emits a {@link TensorFlowCheckpointModel}.
         */
        CHECKPOINT;
    }

    @Config(mandatory=true,description="Path to the protobuf containing the graph.")
    private Path graphPath;

    private GraphDef graphDef;

    @Config(description="Test time batch size.")
    private int testBatchSize = 16;

    @Config(mandatory=true,description="Name of the output operation before the loss.")
    private String outputName;

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

    @Config(description="Saved model format.")
    private TFModelFormat modelFormat = TFModelFormat.TRIBUO_NATIVE;

    @Config(description="Checkpoint output directory.")
    private Path checkpointPath;

    private int trainInvocationCounter = 0;

    /**
     * for olcut.
     */
    private TensorFlowTrainer() {}

    /**
     * Constructs a Trainer for a TensorFlow graph. Stores the model parameters inside the Tribuo model.
     * @param graphPath Path to the graph definition on disk. Must have the necessary targets and placeholders.
     * @param exampleTransformer The example transformer to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputTransformer The output transformer to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param minibatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     * @throws IOException If the graph could not be loaded from the supplied path.
     */
    public TensorFlowTrainer(Path graphPath,
                             String outputName,
                             String initName,
                             GradientOptimiser optimizer,
                             Map<String,Float> gradientParams,
                             ExampleTransformer<T> exampleTransformer,
                             OutputTransformer<T> outputTransformer,
                             int minibatchSize,
                             int epochs,
                             int testBatchSize,
                             int loggingInterval) throws IOException {
        this(graphPath,loadGraphDef(graphPath),outputName,initName,optimizer,gradientParams,exampleTransformer,outputTransformer,minibatchSize,epochs,testBatchSize,loggingInterval,null,TFModelFormat.TRIBUO_NATIVE);
    }

    /**
     * Constructs a Trainer for a TensorFlow graph. Stores the model parameters in a TensorFlow checkpoint.
     * @param graphPath Path to the graph definition on disk. Must have the necessary targets and placeholders.
     * @param exampleTransformer The example transformer to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputTransformer The output transformer to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param minibatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     * @throws IOException If the graph could not be loaded from the supplied path.
     */
    public TensorFlowTrainer(Path graphPath,
                             String outputName,
                             String initName,
                             GradientOptimiser optimizer,
                             Map<String,Float> gradientParams,
                             ExampleTransformer<T> exampleTransformer,
                             OutputTransformer<T> outputTransformer,
                             int minibatchSize,
                             int epochs,
                             int testBatchSize,
                             int loggingInterval,
                             Path checkpointPath) throws IOException {
        this(graphPath,loadGraphDef(graphPath),outputName,initName,optimizer,gradientParams,exampleTransformer,outputTransformer,minibatchSize,epochs,testBatchSize,loggingInterval,checkpointPath,TFModelFormat.CHECKPOINT);
    }

    /**
     * Constructs a Trainer for a TensorFlow graph. Stores the model parameters inside the Tribuo model.
     * @param graphDef The graph definition. Must have the necessary targets and placeholders.
     * @param exampleTransformer The example transformer to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputTransformer The output transformer to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param minibatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     */
    public TensorFlowTrainer(GraphDef graphDef,
                             String outputName,
                             String initName,
                             GradientOptimiser optimizer,
                             Map<String,Float> gradientParams,
                             ExampleTransformer<T> exampleTransformer,
                             OutputTransformer<T> outputTransformer,
                             int minibatchSize,
                             int epochs,
                             int testBatchSize,
                             int loggingInterval) {
        this(null,graphDef,outputName,initName,optimizer,gradientParams,exampleTransformer,outputTransformer,minibatchSize,epochs,testBatchSize,loggingInterval,null,TFModelFormat.TRIBUO_NATIVE);
    }

    /**
     * Constructs a Trainer for a TensorFlow graph. Stores the model parameters in a TensorFlow checkpoint.
     * @param graphDef The graph definition. Must have the necessary targets and placeholders.
     * @param exampleTransformer The example transformer to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputTransformer The output transformer to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param minibatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     */
    public TensorFlowTrainer(GraphDef graphDef,
                             String outputName,
                             String initName,
                             GradientOptimiser optimizer,
                             Map<String,Float> gradientParams,
                             ExampleTransformer<T> exampleTransformer,
                             OutputTransformer<T> outputTransformer,
                             int minibatchSize,
                             int epochs,
                             int testBatchSize,
                             int loggingInterval,
                             Path checkpointPath) {
        this(null,graphDef,outputName,initName,optimizer,gradientParams,exampleTransformer,outputTransformer,minibatchSize,epochs,testBatchSize,loggingInterval,checkpointPath,TFModelFormat.CHECKPOINT);
    }

    /**
     * Constructs a Trainer for a TensorFlow graph. Stores the model parameters inside the Tribuo model.
     * <p>
     * The graph can be closed after the trainer is constructed. Tribuo maintains a copy of the graphdef inside
     * the trainer.
     * @param graph The graph definition. Must have the necessary targets and placeholders.
     * @param exampleTransformer The example transformer to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputTransformer The output transformer to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param minibatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     */
    public TensorFlowTrainer(Graph graph,
                             String outputName,
                             String initName,
                             GradientOptimiser optimizer,
                             Map<String,Float> gradientParams,
                             ExampleTransformer<T> exampleTransformer,
                             OutputTransformer<T> outputTransformer,
                             int minibatchSize,
                             int epochs,
                             int testBatchSize,
                             int loggingInterval) {
        this(null,graph.toGraphDef(),outputName,initName,optimizer,gradientParams,exampleTransformer,outputTransformer,minibatchSize,epochs,testBatchSize,loggingInterval,null,TFModelFormat.TRIBUO_NATIVE);
    }

    /**
     * Constructs a Trainer for a TensorFlow graph. Stores the model parameters in a TensorFlow checkpoint.
     * <p>
     * The graph can be closed after the trainer is constructed. Tribuo maintains a copy of the graphdef inside
     * the trainer.
     * @param graph The graph definition. Must have the necessary targets and placeholders.
     * @param exampleTransformer The example transformer to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputTransformer The output transformer to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param minibatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     */
    public TensorFlowTrainer(Graph graph,
                             String outputName,
                             String initName,
                             GradientOptimiser optimizer,
                             Map<String,Float> gradientParams,
                             ExampleTransformer<T> exampleTransformer,
                             OutputTransformer<T> outputTransformer,
                             int minibatchSize,
                             int epochs,
                             int testBatchSize,
                             int loggingInterval,
                             Path checkpointPath) {
        this(null,graph.toGraphDef(),outputName,initName,optimizer,gradientParams,exampleTransformer,outputTransformer,minibatchSize,epochs,testBatchSize,loggingInterval,checkpointPath,TFModelFormat.CHECKPOINT);
    }

    /**
     * Constructs a Trainer for a TensorFlow graph. One of {@code graphPath} and {@code graph} must be non-null.
     * @param graphPath The path to the graph protobuf. Must have the necessary targets and placeholders.
     * @param graphDef The graph definition protobuf. Must have the necessary targets and placeholders.
     * @param exampleTransformer The example transformer to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputTransformer The output transformer to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param minibatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     * @param checkpointPath The checkpoint path, if using checkpoints.
     * @param modelFormat The model storage format.
     */
    private TensorFlowTrainer(Path graphPath,
                              GraphDef graphDef,
                              String outputName,
                              String initName,
                              GradientOptimiser optimizer,
                              Map<String,Float> gradientParams,
                              ExampleTransformer<T> exampleTransformer,
                              OutputTransformer<T> outputTransformer,
                              int minibatchSize,
                              int epochs,
                              int testBatchSize,
                              int loggingInterval,
                              Path checkpointPath,
                              TFModelFormat modelFormat) {
        if ((graphPath == null) && (graphDef == null)) {
            throw new IllegalArgumentException("Must supply either a GraphDef or a path to a Graph");
        }
        this.graphPath = graphPath;
        this.graphDef = graphDef;
        this.outputName = outputName;
        this.initName = initName;
        this.optimizerEnum = optimizer;
        this.gradientParams = Collections.unmodifiableMap(new HashMap<>(gradientParams));
        this.exampleTransformer = exampleTransformer;
        this.outputTransformer = outputTransformer;
        this.minibatchSize = minibatchSize;
        this.epochs = epochs;
        this.testBatchSize = testBatchSize;
        this.loggingInterval = loggingInterval;
        this.checkpointPath = checkpointPath;
        this.modelFormat = modelFormat;

        // Validate graph.
        try (Graph graph = new Graph()) {
            graph.importGraphDef(graphDef);
            if (graph.operation(initName) == null) {
                throw new IllegalArgumentException("Unable to find the initialization operation, expected an op with name '" + initName + "'");
            }
            Operation outputOp = graph.operation(outputName);
            if (outputOp == null) {
                throw new IllegalArgumentException("Unable to find the output operation, expected an op with name '" + outputName + "'");
            }
            Shape outputShape = outputOp.output(0).shape();
            if (outputShape.numDimensions() != 2) {
                throw new IllegalArgumentException("Expected a 2 dimensional output, found " + Arrays.toString(outputShape.asArray()));
            }
        }
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() throws IOException {
        this.graphDef = loadGraphDef(graphPath);
        if (checkpointPath == null && modelFormat == TFModelFormat.CHECKPOINT) {
            throw new PropertyException("","checkpointPath","Must set 'checkpointPath' when using TFModelFormat.CHECKPOINT");
        }
    }

    /**
     * Loads the graph def protobuf from the path.
     * @param path The path to the protobuf.
     * @return The graph def.
     * @throws IOException If the path could not be loaded.
     */
    private static GraphDef loadGraphDef(Path path) throws IOException {
        try (InputStream stream = new BufferedInputStream(new FileInputStream(path.toFile()))) {
            return GraphDef.parseFrom(stream);
        }
    }

    @Override
    public TensorFlowModel<T> train(Dataset<T> examples) {
        return train(examples,Collections.emptyMap());
    }

    @Override
    public TensorFlowModel<T> train(Dataset<T> examples, Map<String,Provenance> runProvenance) {
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<T> outputInfo = examples.getOutputIDInfo();
        ArrayList<Example<T>> batch = new ArrayList<>();
        Path curCheckpointPath;
        synchronized (this) {
            curCheckpointPath = checkpointPath != null ? Paths.get(checkpointPath.toString(),"invocation-"+trainInvocationCounter) : null;
            trainInvocationCounter++;
        }

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
                    try (TensorMap input = exampleTransformer.transform(batch,featureMap);
                        Tensor target = outputTransformer.transform(batch,outputInfo);
                        Tensor lossTensor = input.feedInto(session.runner())
                                .feed(targetPlaceholder, target)
                                .addTarget(optimizer)
                                .fetch(lossOp)
                                .run().get(0)) {
                        if ((loggingInterval != -1) && (interval % loggingInterval == 0)) {
                            logger.log(Level.INFO, "Training loss = " + ((TFloat32) lossTensor).getFloat());
                        }
                    }
                    interval++;
                }
            }

            // Setup the model serialization infrastructure.
            // **Must** happen before the trainedGraphDef is generated.
            switch (modelFormat) {
                case TRIBUO_NATIVE:
                    TensorFlowUtil.annotateGraph(graph,session);
                    break;
                case CHECKPOINT:
                    session.save(curCheckpointPath.toString());
                    break;
                default:
                    throw new IllegalStateException("Unexpected enum constant " + modelFormat);
            }

            GraphDef trainedGraphDef = graph.toGraphDef();

            ModelProvenance modelProvenance = new ModelProvenance(TensorFlowModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), getProvenance(), runProvenance);
            TensorFlowModel<T> tfModel;

            switch (modelFormat) {
                case TRIBUO_NATIVE:
                    Map<String, TensorFlowUtil.TensorTuple> tensorMap = TensorFlowUtil.serialise(graph,session);
                    tfModel = new TensorFlowNativeModel<>("tf-native-model", modelProvenance, featureMap,
                            outputInfo, trainedGraphDef, tensorMap, testBatchSize, initName, outputOp.op().name(), exampleTransformer, outputTransformer);
                    break;
                case CHECKPOINT:
                    tfModel = new TensorFlowCheckpointModel<>("tf-checkpoint-model", modelProvenance, featureMap,
                            outputInfo, trainedGraphDef, curCheckpointPath.toString(), testBatchSize, initName, outputOp.op().name(), exampleTransformer, outputTransformer);
                    break;
                default:
                    throw new IllegalStateException("Unexpected enum constant " + modelFormat);
            }
            return tfModel;
        } catch (TensorFlowException e) {
            logger.log(Level.SEVERE, "TensorFlow threw an error", e);
            throw new IllegalStateException(e);
        }
    }

    @Override
    public String toString() {
        String path = graphPath==null?"":graphPath.toString();
        String output = "TFTrainer(graphPath="+path+",exampleTransformer="
                +exampleTransformer.toString()+",outputTransformer="+outputTransformer.toString()
                +",minibatchSize="+ minibatchSize +",epochs="+ epochs +",gradientOptimizer="+optimizerEnum
                +",gradientParams="+gradientParams.toString()+",modelFormat="+modelFormat;
        if (modelFormat == TFModelFormat.CHECKPOINT) {
            return output + ",checkpointPath=" + checkpointPath.toString() + ")";
        } else {
            return output + ")";
        }
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TensorFlowTrainerProvenance(this);
    }

    public static final class TensorFlowTrainerProvenance extends SkeletalTrainerProvenance {
        private static final long serialVersionUID = 1L;

        public static final String GRAPH_HASH = "graph-hash";
        public static final String GRAPH_LAST_MOD = "graph-last-modified";

        private final HashProvenance graphHash;
        private final DateTimeProvenance graphLastModified;

        <T extends Output<T>> TensorFlowTrainerProvenance(TensorFlowTrainer<T> host) {
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

        public TensorFlowTrainerProvenance(Map<String,Provenance> map) {
            this(extractTFProvenanceInfo(map));
        }

        private TensorFlowTrainerProvenance(ExtractedInfo info) {
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
            info.instanceValues.put(GRAPH_HASH,ObjectProvenance.checkAndExtractProvenance(info.configuredParameters,GRAPH_HASH,HashProvenance.class, TensorFlowTrainerProvenance.class.getSimpleName()));
            info.instanceValues.put(GRAPH_LAST_MOD,ObjectProvenance.checkAndExtractProvenance(info.configuredParameters,GRAPH_LAST_MOD,DateTimeProvenance.class, TensorFlowTrainerProvenance.class.getSimpleName()));
            return info;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            if (!super.equals(o)) return false;
            TensorFlowTrainerProvenance pairs = (TensorFlowTrainerProvenance) o;
            return graphHash.equals(pairs.graphHash) && graphLastModified.equals(pairs.graphLastModified);
        }

        @Override
        public int hashCode() {
            return Objects.hash(super.hashCode(), graphHash, graphLastModified);
        }
    }
}