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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tensorflow.proto.framework.GraphDef;
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
 * A Tribuo wrapper around a TensorFlow frozen model.
 * <p>
 * The model's serialVersionUID is set to the major Tensorflow version number times 100.
 * <p>
 * N.B. TensorFlow support is experimental and may change without a major version bump.
 */
public final class TensorFlowFrozenExternalModel<T extends Output<T>> extends ExternalModel<T, TensorMap, Tensor> implements Closeable {
    private static final long serialVersionUID = 200L;

    private transient Graph model;

    private transient Session session;

    private final FeatureConverter featureConverter;

    private final OutputConverter<T> outputConverter;

    private final String inputName;

    private final String outputName;

    private TensorFlowFrozenExternalModel(String name, ModelProvenance provenance,
                                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                                          Map<String, Integer> featureMapping,
                                          Graph model, String inputName, String outputName,
                                          FeatureConverter featureConverter, OutputConverter<T> outputConverter) {
        super(name, provenance, featureIDMap, outputIDInfo, outputConverter.generatesProbabilities(), featureMapping);
        this.model = model;
        this.session = new Session(model);
        this.inputName = inputName;
        this.outputName = outputName;
        this.featureConverter = featureConverter;
        this.outputConverter = outputConverter;
    }

    private TensorFlowFrozenExternalModel(String name, ModelProvenance provenance,
                                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                                          int[] featureForwardMapping, int[] featureBackwardMapping,
                                          Graph model, String inputName, String outputName,
                                          FeatureConverter featureConverter, OutputConverter<T> outputConverter) {
        super(name,provenance,featureIDMap,outputIDInfo,featureForwardMapping,featureBackwardMapping,
                outputConverter.generatesProbabilities());
        this.model = model;
        this.session = new Session(model);
        this.inputName = inputName;
        this.outputName = outputName;
        this.featureConverter = featureConverter;
        this.outputConverter = outputConverter;
    }

    @Override
    protected TensorMap convertFeatures(SparseVector input) {
        return featureConverter.convert(input);
    }

    @Override
    protected TensorMap convertFeaturesList(List<SparseVector> input) {
        return featureConverter.convert(input);
    }

    /**
     * Runs the session to make a prediction.
     * <p>
     * Closes the input tensor after the prediction has been made.
     * @param input The input in the external model's format.
     * @return A tensor representing the output.
     */
    @Override
    protected Tensor externalPrediction(TensorMap input) {
        Tensor output = input.feedInto(session.runner()).fetch(outputName).run().get(0);
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
    protected Prediction<T> convertOutput(Tensor output, int numValidFeatures, Example<T> example) {
        Prediction<T> pred = outputConverter.convertToPrediction(output,outputIDInfo,numValidFeatures,example);
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
    protected List<Prediction<T>> convertOutput(Tensor output, int[] numValidFeatures, List<Example<T>> examples) {
        List<Prediction<T>> predictions = outputConverter.convertToBatchPrediction(output,outputIDInfo,numValidFeatures,examples);
        output.close();
        return predictions;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    protected Model<T> copy(String newName, ModelProvenance newProvenance) {
        GraphDef modelBytes = model.toGraphDef();
        Graph newGraph = new Graph();
        newGraph.importGraphDef(modelBytes);
        return new TensorFlowFrozenExternalModel<>(newName,newProvenance,featureIDMap,outputIDInfo,
                featureForwardMapping,featureBackwardMapping,
                newGraph,inputName,outputName,featureConverter, outputConverter);
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
     * Creates a TensorflowFrozenExternalModel by loading in a frozen graph.
     * @param factory The output factory.
     * @param featureMapping The feature mapping between Tribuo's names and the TF integer ids.
     * @param outputMapping The output mapping between Tribuo's names and the TF integer ids.
     * @param inputName The name of the input placeholder.
     * @param outputName The name of the output tensor.
     * @param featureConverter The feature transformation function.
     * @param outputConverter The output transformation function.
     * @param filename The filename to load the graph from.
     * @param <T> The type of the output.
     * @return The TF model wrapped in a Tribuo ExternalModel.
     */
    public static <T extends Output<T>> TensorFlowFrozenExternalModel<T> createTensorflowModel(OutputFactory<T> factory,
                                                                                               Map<String, Integer> featureMapping,
                                                                                               Map<T,Integer> outputMapping,
                                                                                               String inputName,
                                                                                               String outputName,
                                                                                               FeatureConverter featureConverter,
                                                                                               OutputConverter<T> outputConverter,
                                                                                               String filename) {
        try {
            Path path = Paths.get(filename);
            byte[] model = Files.readAllBytes(path);
            Graph graph = new Graph();
            graph.importGraphDef(GraphDef.parseFrom(model));
            URL provenanceLocation = path.toUri().toURL();
            ImmutableFeatureMap featureMap = ExternalModel.createFeatureMap(featureMapping.keySet());
            ImmutableOutputInfo<T> outputInfo = ExternalModel.createOutputInfo(factory,outputMapping);
            OffsetDateTime now = OffsetDateTime.now();
            ExternalTrainerProvenance trainerProvenance = new ExternalTrainerProvenance(provenanceLocation);
            DatasetProvenance datasetProvenance = new ExternalDatasetProvenance("unknown-external-data",factory,false,featureMapping.size(),outputMapping.size());
            ModelProvenance provenance = new ModelProvenance(TensorFlowFrozenExternalModel.class.getName(),now,datasetProvenance,trainerProvenance);
            return new TensorFlowFrozenExternalModel<>("tf-frozen-graph",provenance,featureMap,outputInfo,
                    featureMapping,graph,inputName,outputName,featureConverter, outputConverter);
        } catch (IOException e) {
            throw new IllegalArgumentException("Unable to load model from path " + filename, e);
        }
    }

    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
        GraphDef modelBytes = model.toGraphDef();
        out.writeObject(modelBytes.toByteArray());
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        byte[] modelBytes = (byte[]) in.readObject();
        model = new Graph();
        model.importGraphDef(GraphDef.parseFrom(modelBytes));
        session = new Session(model);
    }

}
