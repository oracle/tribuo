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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.Prediction;
import org.tribuo.interop.ExternalDatasetProvenance;
import org.tribuo.interop.ExternalModel;
import org.tribuo.interop.ExternalTrainerProvenance;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.ModelProvenance;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.Closeable;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.OffsetDateTime;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * A Tribuo wrapper around a Tensorflow frozen model.
 * <p>
 * The model's serialVersionUID is set to the major Tensorflow version number times 100.
 * <p>
 * N.B. Tensorflow support is experimental and may change without a major version bump.
 */
public final class TensorflowExternalModel<T extends Output<T>> extends ExternalModel<T, Tensor<?>, Tensor<?>> implements Closeable {
    private static final long serialVersionUID = 100L;

    private transient Graph model;

    private transient Session session;

    private final ExampleTransformer<T> featureTransformer;

    private final OutputTransformer<T> outputTransformer;

    private final String inputName;

    private final String outputName;

    private TensorflowExternalModel(String name, ModelProvenance provenance,
                                 ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                                 Map<String, Integer> featureMapping,
                                 Graph model, String inputName, String outputName,
                                 ExampleTransformer<T> featureTransformer, OutputTransformer<T> outputTransformer) {
        super(name, provenance, featureIDMap, outputIDInfo, outputTransformer.generatesProbabilities(), featureMapping);
        this.model = model;
        this.session = new Session(model);
        this.inputName = inputName;
        this.outputName = outputName;
        this.featureTransformer = featureTransformer;
        this.outputTransformer = outputTransformer;
    }

    private TensorflowExternalModel(String name, ModelProvenance provenance,
                                    ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                                    int[] featureForwardMapping, int[] featureBackwardMapping,
                                    Graph model, String inputName, String outputName,
                                    ExampleTransformer<T> featureTransformer, OutputTransformer<T> outputTransformer) {
        super(name,provenance,featureIDMap,outputIDInfo,featureForwardMapping,featureBackwardMapping,
                outputTransformer.generatesProbabilities());
        this.model = model;
        this.session = new Session(model);
        this.inputName = inputName;
        this.outputName = outputName;
        this.featureTransformer = featureTransformer;
        this.outputTransformer = outputTransformer;
    }

    @Override
    protected Tensor<?> convertFeatures(SparseVector input) {
        return featureTransformer.transform(input);
    }

    @Override
    protected Tensor<?> convertFeaturesList(List<SparseVector> input) {
        return featureTransformer.transform(input);
    }

    /**
     * Runs the session to make a prediction.
     *
     * Closes the input tensor after the prediction has been made.
     * @param input The input in the external model's format.
     * @return A tensor representing the output.
     */
    @Override
    protected Tensor<?> externalPrediction(Tensor<?> input) {
        Tensor<?> output = session.runner().feed(inputName,input).fetch(outputName).run().get(0);
        input.close();
        return output;
    }

    /**
     * Converts a tensor into a prediction.
     * Closes the output tensor after it's been converted.
     * @param output The output of the external model.
     * @param numValidFeatures The number of valid features in the input.
     * @param example The input example, used to construct the Prediction.
     * @return A {@link Prediction} representing this tensor output.
     */
    @Override
    protected Prediction<T> convertOutput(Tensor<?> output, int numValidFeatures, Example<T> example) {
        Prediction<T> pred = outputTransformer.transformToPrediction(output,outputIDInfo,numValidFeatures,example);
        output.close();
        return pred;
    }

    /**
     * Converts a tensor into a prediction.
     * Closes the output tensor after it's been converted.
     * @param output The output of the external model.
     * @param numValidFeatures An array with the number of valid features in each example.
     * @param examples The input examples, used to construct the Predictions.
     * @return A list of {@link Prediction} representing this tensor output.
     */
    @Override
    protected List<Prediction<T>> convertOutput(Tensor<?> output, int[] numValidFeatures, List<Example<T>> examples) {
        List<Prediction<T>> predictions = outputTransformer.transformToBatchPrediction(output,outputIDInfo,numValidFeatures,examples);
        output.close();
        return predictions;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    protected Model<T> copy(String newName, ModelProvenance newProvenance) {
        byte[] modelBytes = model.toGraphDef();
        Graph newGraph = new Graph();
        newGraph.importGraphDef(modelBytes);
        return new TensorflowExternalModel<>(newName,newProvenance,featureIDMap,outputIDInfo,
                featureForwardMapping,featureBackwardMapping,
                newGraph,inputName,outputName,featureTransformer,outputTransformer);
    }

    @Override
    public void close() {
        if (session != null) {
            session.close();
        }
        if (model != null) {
            model.close();
        }
    }

    /**
     * Creates a TensorflowExternalModel by loading in a frozen graph.
     * @param factory The output factory.
     * @param featureMapping The feature mapping between Tribuo's names and the TF integer ids.
     * @param outputMapping The output mapping between Tribuo's names and the TF integer ids.
     * @param inputName The name of the input placeholder.
     * @param outputName The name of the output tensor.
     * @param featureTransformer The feature transformation function.
     * @param outputTransformer The output transformation function.
     * @param filename The filename to load the graph from.
     * @param <T> The type of the output.
     * @return The TF model wrapped in a Tribuo ExternalModel.
     */
    public static <T extends Output<T>> TensorflowExternalModel<T> createTensorflowModel(OutputFactory<T> factory,
                                                                                      Map<String, Integer> featureMapping,
                                                                                      Map<T,Integer> outputMapping,
                                                                                      String inputName,
                                                                                      String outputName,
                                                                                      ExampleTransformer<T> featureTransformer,
                                                                                      OutputTransformer<T> outputTransformer,
                                                                                      String filename) {
        try {
            Path path = Paths.get(filename);
            byte[] model = Files.readAllBytes(path);
            Graph graph = new Graph();
            graph.importGraphDef(model);
            URL provenanceLocation = path.toUri().toURL();
            ImmutableFeatureMap featureMap = ExternalModel.createFeatureMap(featureMapping.keySet());
            ImmutableOutputInfo<T> outputInfo = ExternalModel.createOutputInfo(factory,outputMapping);
            OffsetDateTime now = OffsetDateTime.now();
            ExternalTrainerProvenance trainerProvenance = new ExternalTrainerProvenance(provenanceLocation);
            DatasetProvenance datasetProvenance = new ExternalDatasetProvenance("unknown-external-data",factory,false,featureMapping.size(),outputMapping.size());
            ModelProvenance provenance = new ModelProvenance(TensorflowExternalModel.class.getName(),now,datasetProvenance,trainerProvenance);
            return new TensorflowExternalModel<>("external-model",provenance,featureMap,outputInfo,
                    featureMapping,graph,inputName,outputName,featureTransformer,outputTransformer);
        } catch (IOException e) {
            throw new IllegalArgumentException("Unable to load model from path " + filename, e);
        }
    }

    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
        byte[] modelBytes = model.toGraphDef();
        out.writeObject(modelBytes);
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        byte[] modelBytes = (byte[]) in.readObject();
        model = new Graph();
        model.importGraphDef(modelBytes);
        session = new Session(model);
    }

}
