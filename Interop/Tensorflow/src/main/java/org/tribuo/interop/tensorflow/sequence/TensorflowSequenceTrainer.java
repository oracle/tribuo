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

package org.tribuo.interop.tensorflow.sequence;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceException;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.HashProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.interop.tensorflow.TensorflowUtil;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.SkeletalTrainerProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.sequence.SequenceDataset;
import org.tribuo.sequence.SequenceExample;
import org.tribuo.sequence.SequenceModel;
import org.tribuo.sequence.SequenceTrainer;
import org.tribuo.util.Util;
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
import java.util.Collections;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A trainer for SequenceModels which use an underlying Tensorflow graph.
 */
public class TensorflowSequenceTrainer<T extends Output<T>> implements SequenceTrainer<T> {
    
    private static final Logger log = Logger.getLogger(TensorflowSequenceTrainer.class.getName());

    @Config(mandatory=true,description="Path to the protobuf containing the Tensorflow graph.")
    protected Path graphPath;

    private byte[] graphDef;

    @Config(mandatory=true,description="Sequence feature extractor.")
    protected SequenceExampleTransformer<T> exampleTransformer;
    @Config(mandatory=true,description="Sequence output extractor.")
    protected SequenceOutputTransformer<T> outputTransformer;

    @Config(description="Minibatch size")
    protected int minibatchSize = 1;
    @Config(description="Number of SGD epochs to run.")
    protected int epochs = 5;
    @Config(description="Logging interval to print the loss.")
    protected int loggingInterval = 100;
    @Config(description="Seed for the RNG.")
    protected long seed = 1;

    @Config(mandatory=true,description="Name of the initialisation operation.")
    protected String initOp;
    @Config(mandatory=true,description="Name of the training operation.")
    protected String trainOp;
    @Config(mandatory=true,description="Name of the loss operation (to inspect the loss).")
    protected String getLossOp;
    @Config(mandatory=true,description="Name of the prediction operation.")
    protected String predictOp;

    protected SplittableRandom rng;

    protected int trainInvocationCounter;

    public TensorflowSequenceTrainer(Path graphPath,
                                     SequenceExampleTransformer<T> exampleTransformer,
                                     SequenceOutputTransformer<T> outputTransformer,
                                     int minibatchSize,
                                     int epochs,
                                     int loggingInterval,
                                     long seed,
                                     String initOp,
                                     String trainOp,
                                     String getLossOp,
                                     String predictOp) throws IOException {
        this.graphPath = graphPath;
        this.exampleTransformer = exampleTransformer;
        this.outputTransformer = outputTransformer;
        this.minibatchSize = minibatchSize;
        this.epochs = epochs;
        this.loggingInterval = loggingInterval;
        this.seed = seed;
        this.initOp = initOp;
        this.trainOp = trainOp;
        this.getLossOp = getLossOp;
        this.predictOp = predictOp;
        postConfig();
    }

    /** Constructor required by olcut config system. **/
    private TensorflowSequenceTrainer() { }

    @Override
    public synchronized void postConfig() throws IOException {
        rng = new SplittableRandom(seed);
        graphDef = Files.readAllBytes(graphPath);
    }

    @Override
    public SequenceModel<T> train(SequenceDataset<T> examples, Map<String,Provenance> runProvenance) {
        // Creates a new RNG, adds one to the invocation count.
        SplittableRandom localRNG;
        TrainerProvenance provenance;
        synchronized(this) {
            localRNG = rng.split();
            provenance = getProvenance();
            trainInvocationCounter++;
        }
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<T> labelMap = examples.getOutputIDInfo();
        ArrayList<SequenceExample<T>> batch = new ArrayList<>();

        int[] indices = Util.randperm(examples.size(), localRNG);

        try (Graph graph = new Graph();
             Session session = new Session(graph)) {
            //
            // Load the graph def into the session.
            graph.importGraphDef(graphDef);
            //
            // Initialise the variables.
            session.runner().addTarget(initOp).run();
            log.info("Initialised the model parameters");
            //
            // Run additional initialization routines, if needed.
            preTrainingHook(session, examples);

            int interval = 0;
            for (int i = 0; i < epochs; i++) {
                log.log(Level.INFO,"Starting epoch " + i);

                // Shuffle the order in which we'll look at examples
                Util.randpermInPlace(indices, localRNG);

                for (int j = 0; j < examples.size(); j += minibatchSize) {
                    batch.clear();
                    for (int k = j; k < (j+ minibatchSize) && k < examples.size(); k++) {
                        int ix = indices[k];
                        batch.add(examples.getExample(ix));
                    }
                    //
                    // Transform examples to tensors
                    Map<String, Tensor<?>> feed = exampleTransformer.encode(batch, featureMap);
                    //
                    // Add supervision
                    feed.putAll(outputTransformer.encode(batch, labelMap));
                    //
                    // Add any additional training hyperparameter values to the feed dict.
                    feed.putAll(getHyperparameterFeed());
                    //
                    // Populate the runner.
                    Session.Runner runner = session.runner();
                    for (Map.Entry<String, Tensor<?>> item : feed.entrySet()) {
                        runner.feed(item.getKey(), item.getValue());
                    }
                    //
                    // Run a training batch.
                    Tensor<?> loss = runner
                            .addTarget(trainOp)
                            .fetch(getLossOp)
                            .run()
                            .get(0);
                    if (interval % loggingInterval == 0) {
                        log.info(String.format("loss %-5.6f [epoch %-2d batch %-4d #(%d - %d)/%d]",
                                loss.floatValue(), i, interval, j, Math.min(examples.size(), j+minibatchSize), examples.size()));
                    }
                    interval++;
                    //
                    // Cleanup: close the tensors.
                    loss.close();
                    for (Tensor<?> tns : feed.values()) {
                        tns.close();
                    }
                }
            }
            
            // This call **must** happen before the trainedGraphDef is generated.
            TensorflowUtil.annotateGraph(graph,session);
            //
            // Generate the trained graph def.
            byte[] trainedGraphDef = graph.toGraphDef();
            Map<String,Object> tensorMap = TensorflowUtil.serialise(graph,session);
            ModelProvenance modelProvenance = new ModelProvenance(TensorflowSequenceModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), provenance, runProvenance);
            return new TensorflowSequenceModel<>(
                    "tf-sequence-model",
                    modelProvenance,
                    featureMap,
                    labelMap,
                    trainedGraphDef,
                    exampleTransformer,
                    outputTransformer,
                    initOp,
                    predictOp,
                    tensorMap
            );

        } catch (TensorFlowException e) {
            log.log(Level.SEVERE, "TensorFlow threw an error", e);
            throw new IllegalStateException(e);
        }
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public String toString() {
        return "TensorflowSequenceTrainer(graphPath="+graphPath.toString()+",exampleTransformer="
                +exampleTransformer.toString()+",outputTransformer="+outputTransformer.toString()
                +",minibatchSize="+ minibatchSize +",epochs="+ epochs +",seed="+seed+")";
    }

    protected void preTrainingHook(Session session, SequenceDataset<T> examples) {}

    protected Map<String, Tensor<?>> getHyperparameterFeed() {
        return Collections.emptyMap();
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TensorflowSequenceTrainerProvenance(this);
    }

    public static class TensorflowSequenceTrainerProvenance extends SkeletalTrainerProvenance {
        private static final long serialVersionUID = 1L;

        public static final String GRAPH_HASH = "graph-hash";
        public static final String GRAPH_LAST_MOD = "graph-last-modified";

        private final StringProvenance graphHash;
        private final DateTimeProvenance graphLastModified;

        <T extends Output<T>> TensorflowSequenceTrainerProvenance(TensorflowSequenceTrainer<T> host) {
            super(host);
            // instance parameters
            this.graphHash = new StringProvenance(GRAPH_HASH,ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE,host.graphPath));
            this.graphLastModified = new DateTimeProvenance(GRAPH_LAST_MOD,OffsetDateTime.ofInstant(Instant.ofEpochMilli(host.graphPath.toFile().lastModified()), ZoneId.systemDefault()));
        }

        public TensorflowSequenceTrainerProvenance(Map<String,Provenance> map) {
            this(extractTFProvenanceInfo(map));
        }

        private TensorflowSequenceTrainerProvenance(ExtractedInfo info) {
            super(info);
            this.graphHash = (StringProvenance) info.instanceValues.get(GRAPH_HASH);
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

            if (info.configuredParameters.containsKey(GRAPH_HASH)) {
                Provenance tmpProv = info.configuredParameters.remove(GRAPH_HASH);
                if (tmpProv instanceof HashProvenance) {
                    info.instanceValues.put(GRAPH_HASH,(HashProvenance) tmpProv);
                } else {
                    throw new ProvenanceException(GRAPH_HASH + " was not of type HashProvenance in class " + info.className);
                }
            } else {
                throw new ProvenanceException("Failed to find " + GRAPH_HASH + " when constructing SkeletalTrainerProvenance");
            }
            if (info.configuredParameters.containsKey(GRAPH_LAST_MOD)) {
                Provenance tmpProv = info.configuredParameters.remove(GRAPH_LAST_MOD);
                if (tmpProv instanceof DateTimeProvenance) {
                    info.instanceValues.put(GRAPH_LAST_MOD,(DateTimeProvenance) tmpProv);
                } else {
                    throw new ProvenanceException(GRAPH_LAST_MOD + " was not of type DateTimeProvenance in class " + info.className);
                }
            } else {
                throw new ProvenanceException("Failed to find " + GRAPH_LAST_MOD + " when constructing SkeletalTrainerProvenance");
            }

            return info;
        }
    }
}