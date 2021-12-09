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
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.proto.framework.ConfigProto;
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
 * Trainer for TensorFlow. Expects the underlying TensorFlow graph to have named placeholders for
 * the inputs, ground truth outputs and a named output operation. The output operation should be
 * before any softmax or sigmoid non-linearities to allow the use of more optimized loss functions.
 * <p>
 * This trainer only works with graphs setup for minibatches. To recover single example training just use a batch size of 1.
 * <p>
 * This trainer uses the serialisation functionality in {@link TensorFlowUtil}, or a TF checkpoint.
 * <p>
 * N.B. TensorFlow support is experimental and may change without a major version bump.
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

    @Config(mandatory=true,description="Feature extractor.")
    private FeatureConverter featureConverter;

    @Config(mandatory=true,description="Response extractor.")
    private OutputConverter<T> outputConverter;

    @Config(description="Training time batch size.")
    private int trainBatchSize = 1;

    @Config(description="Number of SGD epochs to run.")
    private int epochs = 5;

    @Config(description="Logging interval to print out the loss.")
    private int loggingInterval = 100;

    @Config(mandatory=true,description="The gradient optimiser to use.")
    private GradientOptimiser optimiserEnum;

    @Config(mandatory=true,description="The gradient optimiser parameters.")
    private Map<String,Float> gradientParams;

    @Config(description="Saved model format.")
    private TFModelFormat modelFormat = TFModelFormat.TRIBUO_NATIVE;

    @Config(description="Checkpoint output directory.")
    private Path checkpointPath;

    @Config(description="Inter operation thread pool size. -1 uses the default TF value. Tribuo defaults to 1 for deterministic behaviour.")
    private int interOpParallelism = 1;

    @Config(description="Intra operation thread pool size. -1 uses the default TF value. Tribuo defaults to 1 for deterministic behaviour.")
    private int intraOpParallelism = 1;

    private int trainInvocationCounter = 0;

    /**
     * for olcut.
     */
    private TensorFlowTrainer() {}

    /**
     * Constructs a Trainer for a TensorFlow graph. Stores the model parameters inside the Tribuo model.
     * @param graphPath Path to the graph definition on disk. Must have the necessary targets and placeholders.
     * @param outputName The name of the output operation.
     * @param optimiser The gradient optimiser.
     * @param gradientParams The parameters of the gradient optimiser.
     * @param featureConverter The example converter to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputConverter The output converter to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param trainBatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     * @throws IOException If the graph could not be loaded from the supplied path.
     */
    public TensorFlowTrainer(Path graphPath,
                             String outputName,
                             GradientOptimiser optimiser,
                             Map<String,Float> gradientParams,
                             FeatureConverter featureConverter,
                             OutputConverter<T> outputConverter,
                             int trainBatchSize,
                             int epochs,
                             int testBatchSize,
                             int loggingInterval) throws IOException {
        this(graphPath,loadGraphDef(graphPath),outputName,optimiser,gradientParams, featureConverter, outputConverter, trainBatchSize,epochs,testBatchSize,loggingInterval,null,TFModelFormat.TRIBUO_NATIVE);
    }

    /**
     * Constructs a Trainer for a TensorFlow graph. Stores the model parameters in a TensorFlow checkpoint.
     * @param graphPath Path to the graph definition on disk. Must have the necessary targets and placeholders.
     * @param outputName The name of the output operation.
     * @param optimiser The gradient optimiser.
     * @param gradientParams The parameters of the gradient optimiser.
     * @param featureConverter The example converter to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputConverter The output converter to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param trainBatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     * @param checkpointPath The path to save out the TensorFlow checkpoint.
     * @throws IOException If the graph could not be loaded from the supplied path.
     */
    public TensorFlowTrainer(Path graphPath,
                             String outputName,
                             GradientOptimiser optimiser,
                             Map<String,Float> gradientParams,
                             FeatureConverter featureConverter,
                             OutputConverter<T> outputConverter,
                             int trainBatchSize,
                             int epochs,
                             int testBatchSize,
                             int loggingInterval,
                             Path checkpointPath) throws IOException {
        this(graphPath,loadGraphDef(graphPath),outputName,optimiser,gradientParams, featureConverter, outputConverter, trainBatchSize,epochs,testBatchSize,loggingInterval,checkpointPath,TFModelFormat.CHECKPOINT);
    }

    /**
     * Constructs a Trainer for a TensorFlow graph. Stores the model parameters inside the Tribuo model.
     * @param graphDef The graph definition. Must have the necessary targets and placeholders.
     * @param outputName The name of the output operation.
     * @param optimiser The gradient optimiser.
     * @param gradientParams The parameters of the gradient optimiser.
     * @param featureConverter The example converter to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputConverter The output converter to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param trainBatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     */
    public TensorFlowTrainer(GraphDef graphDef,
                             String outputName,
                             GradientOptimiser optimiser,
                             Map<String,Float> gradientParams,
                             FeatureConverter featureConverter,
                             OutputConverter<T> outputConverter,
                             int trainBatchSize,
                             int epochs,
                             int testBatchSize,
                             int loggingInterval) {
        this(null,graphDef,outputName,optimiser,gradientParams, featureConverter, outputConverter, trainBatchSize,epochs,testBatchSize,loggingInterval,null,TFModelFormat.TRIBUO_NATIVE);
    }

    /**
     * Constructs a Trainer for a TensorFlow graph. Stores the model parameters in a TensorFlow checkpoint.
     * @param graphDef The graph definition. Must have the necessary targets and placeholders.
     * @param outputName The name of the output operation.
     * @param optimiser The gradient optimiser.
     * @param gradientParams The parameters of the gradient optimiser.
     * @param featureConverter The example converter to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputConverter The output converter to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param trainBatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     * @param checkpointPath The path to save out the TensorFlow checkpoint.
     */
    public TensorFlowTrainer(GraphDef graphDef,
                             String outputName,
                             GradientOptimiser optimiser,
                             Map<String,Float> gradientParams,
                             FeatureConverter featureConverter,
                             OutputConverter<T> outputConverter,
                             int trainBatchSize,
                             int epochs,
                             int testBatchSize,
                             int loggingInterval,
                             Path checkpointPath) {
        this(null,graphDef,outputName,optimiser,gradientParams, featureConverter, outputConverter, trainBatchSize,epochs,testBatchSize,loggingInterval,checkpointPath,TFModelFormat.CHECKPOINT);
    }

    /**
     * Constructs a Trainer for a TensorFlow graph. Stores the model parameters inside the Tribuo model.
     * <p>
     * The graph can be closed after the trainer is constructed. Tribuo maintains a copy of the graphdef inside
     * the trainer.
     * @param graph The graph definition. Must have the necessary targets and placeholders.
     * @param outputName The name of the output operation.
     * @param optimiser The gradient optimiser.
     * @param gradientParams The parameters of the gradient optimiser.
     * @param featureConverter The example converter to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputConverter The output converter to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param trainBatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     */
    public TensorFlowTrainer(Graph graph,
                             String outputName,
                             GradientOptimiser optimiser,
                             Map<String,Float> gradientParams,
                             FeatureConverter featureConverter,
                             OutputConverter<T> outputConverter,
                             int trainBatchSize,
                             int epochs,
                             int testBatchSize,
                             int loggingInterval) {
        this(null,graph.toGraphDef(),outputName,optimiser,gradientParams, featureConverter, outputConverter, trainBatchSize,epochs,testBatchSize,loggingInterval,null,TFModelFormat.TRIBUO_NATIVE);
    }

    /**
     * Constructs a Trainer for a TensorFlow graph. Stores the model parameters in a TensorFlow checkpoint.
     * <p>
     * The graph can be closed after the trainer is constructed. Tribuo maintains a copy of the graphdef inside
     * the trainer.
     * @param graph The graph definition. Must have the necessary targets and placeholders.
     * @param outputName The name of the output operation.
     * @param optimiser The gradient optimiser.
     * @param gradientParams The parameters of the gradient optimiser.
     * @param featureConverter The example converter to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputConverter The output converter to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param trainBatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     * @param checkpointPath The checkpoint path, if using checkpoints.
     */
    public TensorFlowTrainer(Graph graph,
                             String outputName,
                             GradientOptimiser optimiser,
                             Map<String,Float> gradientParams,
                             FeatureConverter featureConverter,
                             OutputConverter<T> outputConverter,
                             int trainBatchSize,
                             int epochs,
                             int testBatchSize,
                             int loggingInterval,
                             Path checkpointPath) {
        this(null,graph.toGraphDef(),outputName,optimiser,gradientParams, featureConverter, outputConverter, trainBatchSize,epochs,testBatchSize,loggingInterval,checkpointPath,TFModelFormat.CHECKPOINT);
    }

    /**
     * Constructs a Trainer for a TensorFlow graph. One of {@code graphPath} and {@code graph} must be non-null.
     * @param graphPath The path to the graph protobuf. Must have the necessary targets and placeholders.
     * @param graphDef The graph definition protobuf. Must have the necessary targets and placeholders.
     * @param outputName The name of the output operation.
     * @param optimiser The gradient optimiser.
     * @param gradientParams The parameters of the gradient optimiser.
     * @param featureConverter The example converter to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputConverter The output converter to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param trainBatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     * @param loggingInterval The logging interval. Set to -1 to quiesce the loss level logging.
     * @param checkpointPath The checkpoint path, if using checkpoints.
     * @param modelFormat The model storage format.
     */
    private TensorFlowTrainer(Path graphPath,
                              GraphDef graphDef,
                              String outputName,
                              GradientOptimiser optimiser,
                              Map<String,Float> gradientParams,
                              FeatureConverter featureConverter,
                              OutputConverter<T> outputConverter,
                              int trainBatchSize,
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
        this.optimiserEnum = optimiser;
        this.gradientParams = Collections.unmodifiableMap(new HashMap<>(gradientParams));
        this.featureConverter = featureConverter;
        this.outputConverter = outputConverter;
        this.trainBatchSize = trainBatchSize;
        this.epochs = epochs;
        this.testBatchSize = testBatchSize;
        this.loggingInterval = loggingInterval;
        this.checkpointPath = checkpointPath;
        this.modelFormat = modelFormat;

        validateGraph(false);
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
        validateGraph(true);
    }

    /**
     * Validates that the graph has the appropriate input, output and initialization operations.
     * <p>
     * Throws IllegalArgumentException or PropertyException if the graph is invalid.
     * @param throwPropertyException If true throw PropertyException instead of IllegalArgumentException.
     */
    private void validateGraph(boolean throwPropertyException) {
        try (Graph graph = new Graph()) {
            graph.importGraphDef(graphDef);
            for (String inputName : featureConverter.inputNamesSet()) {
                if (graph.operation(inputName) == null) {
                    String msg = "Unable to find an input operation, expected an op with name '" + inputName + "'";
                    if (throwPropertyException) {
                        throw new PropertyException("","featureConverter",msg);
                    } else {
                        throw new IllegalArgumentException(msg);
                    }
                }
            }
            Operation outputOp = graph.operation(outputName);
            if (outputOp == null) {
                String msg = "Unable to find the output operation, expected an op with name '" + outputName + "'";
                if (throwPropertyException) {
                    throw new PropertyException("","outputName",msg);
                } else {
                    throw new IllegalArgumentException(msg);
                }
            }
            Shape outputShape = outputOp.output(0).shape();
            if (outputShape.numDimensions() != 2) {
                String msg = "Expected a 2 dimensional output, found " + Arrays.toString(outputShape.asArray());
                if (throwPropertyException) {
                    throw new PropertyException("","outputName",msg);
                } else {
                    throw new IllegalArgumentException(msg);
                }
            }
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
        return(train(examples, runProvenance, INCREMENT_INVOCATION_COUNT));
    }

    @Override
    public TensorFlowModel<T> train(Dataset<T> examples, Map<String,Provenance> runProvenance, int invocationCount) {
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<T> outputInfo = examples.getOutputIDInfo();
        ArrayList<Example<T>> batch = new ArrayList<>();
        Path curCheckpointPath;
        synchronized (this) {
            if(invocationCount != INCREMENT_INVOCATION_COUNT) {
                setInvocationCount(invocationCount);
            }
            curCheckpointPath = checkpointPath != null ? Paths.get(checkpointPath.toString(),"invocation-"+trainInvocationCounter, "tribuo") : null;
            trainInvocationCounter++;
        }

        ConfigProto.Builder configBuilder = ConfigProto.newBuilder();
        if (interOpParallelism > -1) {
            configBuilder.setInterOpParallelismThreads(interOpParallelism);
        }
        if (intraOpParallelism > -1) {
            configBuilder.setIntraOpParallelismThreads(intraOpParallelism);
        }
        ConfigProto config = configBuilder.build();

        try (Graph graph = new Graph();
             Session session = new Session(graph,config)) {
            // Load in the graph definition
            graph.importGraphDef(graphDef);
            Ops tf = Ops.create(graph).withName("tribuo-internal");

            // Lookup output op
            Operand<TNumber> intermediateOutputOp = graph.operation(outputName).output(0);

            // Validate that the output op is the right shape
            Shape outputShape = intermediateOutputOp.shape();
            Shape expectedShape = Shape.of(trainBatchSize,outputInfo.size());
            if (!outputShape.isCompatibleWith(expectedShape)) {
                throw new IllegalArgumentException("Incompatible output shape, expected " + expectedShape.toString() + " found " + outputShape.toString());
            }

            // Add target placeholder
            Placeholder<? extends TNumber> targetPlaceholder = tf.placeholder(TFloat32.class,Placeholder.shape(Shape.of(trainBatchSize,outputInfo.size())));

            // Add loss, optimiser and output
            Op outputOp = outputConverter.outputTransformFunction().apply(tf,intermediateOutputOp);
            Operand<TNumber> lossOp = outputConverter.loss().apply(tf,new Pair<>(targetPlaceholder,intermediateOutputOp));
            Op optimiser = optimiserEnum.applyOptimiser(graph,lossOp,gradientParams);

            // Initalise all the things
            session.initialize();

            logger.info("Initialised the model parameters");

            int interval = 0;
            for (int i = 0; i < epochs; i++) {
                logger.log(Level.INFO,"Starting epoch " + i);
                for (int j = 0; j < examples.size(); j += trainBatchSize) {
                    batch.clear();
                    for (int k = j; k < (j+ trainBatchSize) && k < examples.size(); k++) {
                        batch.add(examples.getExample(k));
                    }
                    try (TensorMap input = featureConverter.convert(batch,featureMap);
                         Tensor target = outputConverter.convertToTensor(batch,outputInfo);
                         Tensor lossTensor = input.feedInto(session.runner())
                                .feed(targetPlaceholder, target)
                                .addTarget(optimiser)
                                .fetch(lossOp)
                                .run().get(0)) {
                        if ((loggingInterval != -1) && (interval % loggingInterval == 0)) {
                            logger.log(Level.INFO, "Training loss at itr " + interval + " = " + ((TFloat32) lossTensor).getFloat());
                        }
                    }
                    interval++;
                }
            }

            // Setup the model serialization infrastructure.
            // **Must** happen before the trainedGraphDef is generated.

            // We unconditionally annotate the Graph for Tribuo's serialization
            TensorFlowUtil.annotateGraph(graph,session);

            // If it's a checkpoint we also save it out.
            if (modelFormat == TFModelFormat.CHECKPOINT) {
                session.save(curCheckpointPath.toString());
            }

            GraphDef trainedGraphDef = graph.toGraphDef();

            ModelProvenance modelProvenance = new ModelProvenance(TensorFlowModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), getProvenance(), runProvenance);
            TensorFlowModel<T> tfModel;

            switch (modelFormat) {
                case TRIBUO_NATIVE:
                    Map<String, TensorFlowUtil.TensorTuple> tensorMap = TensorFlowUtil.extractMarshalledVariables(graph,session);
                    tfModel = new TensorFlowNativeModel<>("tf-native-model", modelProvenance, featureMap,
                            outputInfo, trainedGraphDef, tensorMap, testBatchSize, outputOp.op().name(), featureConverter, outputConverter);
                    break;
                case CHECKPOINT:
                    tfModel = new TensorFlowCheckpointModel<>("tf-checkpoint-model", modelProvenance, featureMap,
                            outputInfo, trainedGraphDef, curCheckpointPath.getParent().toString(), curCheckpointPath.getFileName().toString(), testBatchSize, outputOp.op().name(), featureConverter, outputConverter);
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
        String output = "TFTrainer(graphPath="+path+",exampleConverter="
                + featureConverter.toString()+",outputConverter="+ outputConverter.toString()
                +",minibatchSize="+ trainBatchSize +",epochs="+ epochs +",gradientOptimizer="+ optimiserEnum
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
    public void setInvocationCount(int invocationCount) {
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }

        this.trainInvocationCounter = invocationCount;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TensorFlowTrainerProvenance(this);
    }

    /**
     * Provenance for {@link TensorFlowTrainer}.
     */
    public static final class TensorFlowTrainerProvenance extends SkeletalTrainerProvenance {
        private static final long serialVersionUID = 1L;

        /**
         * The name of the provenance field containing the graph hash.
         */
        public static final String GRAPH_HASH = "graph-hash";
        /**
         * The name of the provenance field containing the graph modified timestamp.
         */
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

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
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
