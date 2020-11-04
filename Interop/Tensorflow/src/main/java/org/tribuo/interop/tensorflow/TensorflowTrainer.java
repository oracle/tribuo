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
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.SkeletalTrainerProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlowException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
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
 * This trainer uses the serialisation functionality in {@link TensorflowUtil}, as opposed to a SavedModel or a checkpoint.
 * <p>
 * N.B. Tensorflow support is experimental and may change without a major version bump.
 */
public final class TensorflowTrainer<T extends Output<T>> implements Trainer<T> {

    private static final Logger logger = Logger.getLogger(TensorflowTrainer.class.getName());

    public static final String TARGET = "target";
    public static final String TRAIN = "train";
    public static final String TRAINING_LOSS = "training_loss";
    public static final String EPOCH = "epoch";
    public static final String IS_TRAINING = "is_training";
    public static final String INIT = "init";

    @Config(mandatory=true,description="Path to the protobuf containing the graph.")
    private Path graphPath;

    private byte[] graphDef;

    @Config(description="Test time batch size.")
    private int testBatchSize = 16;

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

    private int trainInvocationCounter = 0;

    /**
     * for olcut.
     */
    private TensorflowTrainer() {}

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
    public TensorflowTrainer(Path graphPath, ExampleTransformer<T> exampleTransformer, OutputTransformer<T> outputTransformer, int minibatchSize, int epochs, int testBatchSize) throws IOException {
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
     * @param graphDef The graph definition as a byte array. Must have the targets and placeholders specified above.
     * @param exampleTransformer The example transformer to convert a Tribuo {@link Example} into a {@link Tensor}.
     * @param outputTransformer The output transformer to convert a Tribuo {@link Output} into a {@link Tensor} and back. This encodes the output type.
     * @param minibatchSize The minibatch size to use in training.
     * @param epochs The number of SGD epochs to run.
     * @param testBatchSize The minibatch size to use at test time.
     */
    public TensorflowTrainer(byte[] graphDef, ExampleTransformer<T> exampleTransformer, OutputTransformer<T> outputTransformer, int minibatchSize, int epochs, int testBatchSize) {
        this.graphPath = null;
        this.graphDef = graphDef;
        this.exampleTransformer = exampleTransformer;
        this.outputTransformer = outputTransformer;
        this.minibatchSize = minibatchSize;
        this.epochs = epochs;
        this.testBatchSize = testBatchSize;
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() throws IOException {
        graphDef = Files.readAllBytes(graphPath);
    }

    @Override
    public Model<T> train(Dataset<T> examples, Map<String,Provenance> runProvenance) {
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<T> outputInfo = examples.getOutputIDInfo();
        ArrayList<Example<T>> batch = new ArrayList<>();
        trainInvocationCounter++;

        try (Graph graph = new Graph();
             Session session = new Session(graph);
             Tensor<?> isTraining = Tensor.create(true)) {
            // Load in the graph definition
            graph.importGraphDef(graphDef);

            // Initialises the parameters.
            session.runner().addTarget(INIT).run();
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
                            .feed(TARGET, target)
                            .feed(EPOCH, epoch)
                            .feed(IS_TRAINING, isTraining)
                            .addTarget(TRAIN)
                            .fetch(TRAINING_LOSS)
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

            //System.out.println("After training");
            //TensorflowModel.print(session);

            // This call **must** happen before the trainedGraphDef is generated.
            TensorflowUtil.annotateGraph(graph,session);

            byte[] trainedGraphDef = graph.toGraphDef();

            Map<String,Object> tensorMap = TensorflowUtil.serialise(graph,session);

            ModelProvenance modelProvenance = new ModelProvenance(TensorflowModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), getProvenance(), runProvenance);
            TensorflowModel<T> tfModel = new TensorflowModel<>("tf-model", modelProvenance, featureMap,
                    outputInfo, trainedGraphDef, tensorMap, testBatchSize, exampleTransformer, outputTransformer);

            return tfModel;
        } catch (TensorFlowException e) {
            logger.log(Level.SEVERE, "TensorFlow threw an error", e);
            throw new IllegalStateException(e);
        }
    }

    @Override
    public String toString() {
        String path = graphPath==null?"":graphPath.toString();
        return "TensorflowTrainer(graphPath="+path+",exampleTransformer="
                +exampleTransformer.toString()+",outputTransformer="+outputTransformer.toString()
                +",minibatchSize="+ minibatchSize +",epochs="+ epochs +")";
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TensorflowTrainerProvenance(this);
    }

    public static final class TensorflowTrainerProvenance extends SkeletalTrainerProvenance {
        private static final long serialVersionUID = 1L;

        public static final String GRAPH_HASH = "graph-hash";
        public static final String GRAPH_LAST_MOD = "graph-last-modified";

        private final HashProvenance graphHash;
        private final DateTimeProvenance graphLastModified;

        <T extends Output<T>> TensorflowTrainerProvenance(TensorflowTrainer<T> host) {
            super(host);
            // instance parameters
            if (host.graphPath != null) {
                this.graphHash = new HashProvenance(DEFAULT_HASH_TYPE,GRAPH_HASH,ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE,host.graphPath));
                this.graphLastModified = new DateTimeProvenance(GRAPH_LAST_MOD, OffsetDateTime.ofInstant(Instant.ofEpochMilli(host.graphPath.toFile().lastModified()), ZoneId.systemDefault()));
            } else {
                this.graphHash = new HashProvenance(DEFAULT_HASH_TYPE,GRAPH_HASH,hashArray(DEFAULT_HASH_TYPE,host.graphDef));
                this.graphLastModified = new DateTimeProvenance(GRAPH_LAST_MOD, OffsetDateTime.now());
            }
        }

        public TensorflowTrainerProvenance(Map<String,Provenance> map) {
            this(extractTFProvenanceInfo(map));
        }

        private TensorflowTrainerProvenance(ExtractedInfo info) {
            super(info);
            this.graphHash = (HashProvenance) info.instanceValues.get(GRAPH_HASH);
            this.graphLastModified = (DateTimeProvenance) info.instanceValues.get(GRAPH_LAST_MOD);
        }

        /**
         * Hashes a byte array using the specified {@link com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil.HashType}.
         * @param hashType The type of hash to perform.
         * @param input The input array.
         * @return A hexadecimal string representation of the hash.
         */
        private static String hashArray(com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil.HashType hashType, byte[] input) {
            java.security.MessageDigest md = hashType.getDigest();
            md.update(input);
            return ProvenanceUtil.bytesToHexString(md.digest());
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
            info.instanceValues.put(GRAPH_HASH,ObjectProvenance.checkAndExtractProvenance(map,GRAPH_HASH,HashProvenance.class,TensorflowTrainerProvenance.class.getSimpleName()));
            info.instanceValues.put(GRAPH_LAST_MOD,ObjectProvenance.checkAndExtractProvenance(map,GRAPH_LAST_MOD,DateTimeProvenance.class,TensorflowTrainerProvenance.class.getSimpleName()));
            return info;
        }
    }
}
