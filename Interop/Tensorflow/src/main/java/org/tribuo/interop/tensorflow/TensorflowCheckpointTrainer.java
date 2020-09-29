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

package org.tribuo.interop.tensorflow;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.HashProvenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.interop.tensorflow.TensorflowTrainer.TensorflowTrainerProvenance;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.SkeletalTrainerProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlowException;
import org.tensorflow.Tensors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Trainer for Tensorflow. Expects the underlying Tensorflow graph to have specific placeholders and
 * targets listed below.
 *
 * <ul>
 * <li>{@link TensorflowModel#INPUT_NAME} - the input minibatch.</li>
 * <li>{@link TensorflowModel#OUTPUT_NAME} - the predicted output.</li>
 * <li>{@link TensorflowTrainer#TARGET} - the output to predict.</li>
 * <li>{@link TensorflowTrainer#TRAIN} - the train function to run (usually a single step of SGD).</li>
 * <li>{@link TensorflowTrainer#TRAINING_LOSS} - the loss tensor to extract for logging.</li>
 * <li>{@link TensorflowTrainer#EPOCH} - the current epoch number, used for gradient scaling.</li>
 * <li>{@link TensorflowTrainer#IS_TRAINING} - a boolean placeholder to turn on dropout or other training specific functionality.</li>
 * <li>{@link TensorflowTrainer#INIT} - the function to initialise the graph.</li>
 * </ul>
 *
 * This trainer only works with graphs setup for minibatches. To recover single example training just use a batch size of 1.
 * <p>
 * This trainer uses the native Tensorflow serialisation functionality and saves to a checkpoint on disk. It's much more
 * fragile than the {@link TensorflowTrainer}.
 * </p>
 * <p>
 * N.B. Tensorflow support is experimental and may change without a major version bump.
 */
public final class TensorflowCheckpointTrainer<T extends Output<T>> implements Trainer<T> {

    private static final Logger logger = Logger.getLogger(TensorflowCheckpointTrainer.class.getName());

    public static final String MODEL_FILENAME = "model";

    @Config(mandatory=true,description="Path to the protobuf containing the graph.")
    private Path graphPath;

    private byte[] graphDef;

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

    @Config(description="Path to write out the checkpoints.")
    private Path checkpointRootPath = Paths.get("/tmp/");

    private int trainInvocationCounter = 0;

    /**
     * for olcut.
     */
    private TensorflowCheckpointTrainer() {}

    /**
     * Builds a trainer using the supplied graph and arguments.
     * @param graphPath The graph to load.
     * @param checkpointRootPath The checkpoint path to save to.
     * @param exampleTransformer The feature transformer.
     * @param outputTransformer The output transformer.
     * @param minibatchSize The training batch size.
     * @param epochs The number of training epochs.
     * @throws IOException If the graph failed to load.
     */
    public TensorflowCheckpointTrainer(Path graphPath, Path checkpointRootPath, ExampleTransformer<T> exampleTransformer, OutputTransformer<T> outputTransformer, int minibatchSize, int epochs) throws IOException {
        this.graphPath = graphPath;
        this.checkpointRootPath = checkpointRootPath;
        this.exampleTransformer = exampleTransformer;
        this.outputTransformer = outputTransformer;
        this.minibatchSize = minibatchSize;
        this.epochs = epochs;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() throws IOException {
        graphDef = Files.readAllBytes(graphPath);
    }

    @Override
    public Model<T> train(Dataset<T> examples, Map<String, Provenance> runProvenance) {
        Path checkpointPath;
        try {
            checkpointPath = Files.createTempDirectory(checkpointRootPath,"tensorflow-checkpoint");
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Failed to create checkpoint directory at path " + checkpointRootPath,e);
            throw new IllegalStateException("Failed to create checkpoint directory at path " + checkpointRootPath,e);
        }
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<T> outputInfo = examples.getOutputIDInfo();
        ArrayList<Example<T>> batch = new ArrayList<>();

        trainInvocationCounter++;

        try (Graph graph = new Graph();
             Session session = new Session(graph);
             Tensor<?> isTraining = Tensor.create(true);
             Tensor<String> checkpointPathTensor = Tensors.create(checkpointPath.toString()+"/"+MODEL_FILENAME) ) {
            // Load in the graph definition
            graph.importGraphDef(graphDef);

            // Initialises the parameters.
            session.runner().addTarget(TensorflowTrainer.INIT).run();
            logger.info("Initialised the model parameters");

            int interval = 0;
            for (int i = 0; i < epochs; i++) {
                logger.log(Level.INFO,"Starting epoch " + i);
                Tensor<?> epoch = Tensor.create(i);
                for (int j = 0; j < examples.size(); j += minibatchSize) {
                    batch.clear();
                    for (int k = j; k < (j+ minibatchSize) && k < examples.size(); k++) {
                        batch.add(examples.getExample(k));
                    }
                    //logger.info("Batch = " + batch.size());
                    Tensor<?> input = exampleTransformer.transform(batch,featureMap);
                    Tensor<?> target = outputTransformer.transform(batch,outputInfo);
                    Tensor<?> loss = session.runner()
                            .feed(TensorflowModel.INPUT_NAME, input)
                            .feed(TensorflowTrainer.TARGET, target)
                            .feed(TensorflowTrainer.EPOCH, epoch)
                            .feed(TensorflowTrainer.IS_TRAINING, isTraining)
                            .addTarget(TensorflowTrainer.TRAIN)
                            .fetch(TensorflowTrainer.TRAINING_LOSS)
                            .run().get(0);
                    if (interval % loggingInterval == 0) {
                        logger.log(Level.INFO, "Training loss = " + loss.floatValue());
                    }
                    input.close();
                    target.close();
                    loss.close();
                    interval++;
                }
                epoch.close();
            }

            session.runner().feed("save/Const", checkpointPathTensor).addTarget("save/control_dependency").run();

            byte[] trainedGraphDef = graph.toGraphDef();

            ModelProvenance modelProvenance = new ModelProvenance(TensorflowCheckpointModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), getProvenance(), runProvenance);
            TensorflowCheckpointModel<T> tfModel = new TensorflowCheckpointModel<>("tf-model", modelProvenance, featureMap,
                    outputInfo, trainedGraphDef, checkpointPath.toString(), exampleTransformer, outputTransformer);

            return tfModel;
        } catch (TensorFlowException e) {
            logger.log(Level.SEVERE, "TensorFlow threw an error", e);
            throw new IllegalStateException(e);
        }
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public String toString() {
        return "TensorflowCheckpointTrainer(graphPath="+graphPath.toString()
                +",checkpointRootPath="+checkpointRootPath.toString()+",exampleTransformer="
                +exampleTransformer.toString()+",outputTransformer"+outputTransformer.toString()
                +",minibatchSize="+ minibatchSize +",epochs="+ epochs +")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TensorflowCheckpointTrainerProvenance(this);
    }

    public static final class TensorflowCheckpointTrainerProvenance extends SkeletalTrainerProvenance {
        private static final long serialVersionUID = 1L;

        public static final String GRAPH_HASH = "graph-hash";
        public static final String GRAPH_LAST_MOD = "graph-last-modified";

        private final HashProvenance graphHash;
        private final DateTimeProvenance graphLastModified;

        <T extends Output<T>> TensorflowCheckpointTrainerProvenance(TensorflowCheckpointTrainer<T> host) {
            super(host);
            // instance parameters
            this.graphHash = new HashProvenance(DEFAULT_HASH_TYPE,GRAPH_HASH,ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE,host.graphPath));
            this.graphLastModified = new DateTimeProvenance(GRAPH_LAST_MOD, OffsetDateTime.ofInstant(Instant.ofEpochMilli(host.graphPath.toFile().lastModified()), ZoneId.systemDefault()));
        }

        public TensorflowCheckpointTrainerProvenance(Map<String,Provenance> map) {
            this(extractTFProvenanceInfo(map));
        }

        private TensorflowCheckpointTrainerProvenance(ExtractedInfo info) {
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
            info.instanceValues.put(GRAPH_HASH, ObjectProvenance.checkAndExtractProvenance(map,GRAPH_HASH,HashProvenance.class, TensorflowTrainerProvenance.class.getSimpleName()));
            info.instanceValues.put(GRAPH_LAST_MOD,ObjectProvenance.checkAndExtractProvenance(map,GRAPH_LAST_MOD,DateTimeProvenance.class,TensorflowTrainerProvenance.class.getSimpleName()));
            return info;
        }
    }
}
