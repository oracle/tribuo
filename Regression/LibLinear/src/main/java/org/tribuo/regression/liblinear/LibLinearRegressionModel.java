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

package org.tribuo.regression.liblinear;

import ai.onnx.proto.OnnxMl;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.ONNXExportable;
import org.tribuo.Prediction;
import org.tribuo.common.liblinear.LibLinearModel;
import org.tribuo.common.liblinear.LibLinearTrainer;
import org.tribuo.common.liblinear.protos.LibLinearModelProto;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.ImmutableRegressionInfo;
import org.tribuo.regression.Regressor;
import org.tribuo.util.onnx.ONNXContext;
import org.tribuo.util.onnx.ONNXInitializer;
import org.tribuo.util.onnx.ONNXNode;
import org.tribuo.util.onnx.ONNXOperators;
import org.tribuo.util.onnx.ONNXPlaceholder;
import org.tribuo.util.onnx.ONNXRef;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.logging.Logger;

/**
 * A {@link Model} which wraps a LibLinear-java model.
 * <p>
 * It disables the LibLinear debug output as it's very chatty.
 * <p>
 * It contains an independent liblinear model for each regression dimension.
 * <p>
 * See:
 * <pre>
 * Fan RE, Chang KW, Hsieh CJ, Wang XR, Lin CJ.
 * "LIBLINEAR: A library for Large Linear Classification"
 * Journal of Machine Learning Research, 2008.
 * </pre>
 * and for the original algorithm:
 * <pre>
 * Cortes C, Vapnik V.
 * "Support-Vector Networks"
 * Machine Learning, 1995.
 * </pre>
 */
public class LibLinearRegressionModel extends LibLinearModel<Regressor> implements ONNXExportable {
    private static final long serialVersionUID = 2L;

    private static final Logger logger = Logger.getLogger(LibLinearRegressionModel.class.getName());

    private final String[] dimensionNames;

    // Not final as it doesn't exist in 4.0 or 4.1 and so must be created on deserialization.
    private int[] mapping;

    LibLinearRegressionModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputInfo, List<de.bwaldvogel.liblinear.Model> models) {
        super(name, description, featureIDMap, outputInfo, false, models);
        this.dimensionNames = Regressor.extractNames(outputInfo);
        this.mapping = ((ImmutableRegressionInfo) outputInfo).getIDtoNaturalOrderMapping();
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static LibLinearRegressionModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (!"org.tribuo.regression.liblinear.LibLinearRegressionModel".equals(className)) {
            throw new IllegalStateException("Invalid protobuf, this class can only deserialize LibLinearRegressionModel");
        }
        LibLinearModelProto proto = message.unpack(LibLinearModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Regressor.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a regression domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Regressor> outputDomain = (ImmutableOutputInfo<Regressor>) carrier.outputDomain();

        if (proto.getModelsCount() != outputDomain.size()) {
            throw new IllegalStateException("Invalid protobuf, expected " + outputDomain.size() + " model, found " + proto.getModelsCount());
        }
        try {
            List<de.bwaldvogel.liblinear.Model> models = new ArrayList<>();
            for (ByteString modelArray : proto.getModelsList()) {
                ByteArrayInputStream bais = new ByteArrayInputStream(modelArray.toByteArray());
                ObjectInputStream ois = new ObjectInputStream(bais);
                de.bwaldvogel.liblinear.Model model = (de.bwaldvogel.liblinear.Model) ois.readObject();
                ois.close();
                models.add(model);
            }
            return new LibLinearRegressionModel(carrier.name(),carrier.provenance(),carrier.featureDomain(),outputDomain,Collections.unmodifiableList(models));
        } catch (IOException | ClassNotFoundException e) {
            throw new IllegalStateException("Invalid protobuf, failed to deserialize liblinear model", e);
        }
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        FeatureNode[] features = LibLinearTrainer.exampleToNodes(example, featureIDMap, null);
        // Bias feature is always set
        if (features.length == 1) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }

        double[] scores = new double[models.get(0).getNrClass()];
        double[] regressedValues = new double[models.size()];

        // Map through the id -> regressor dimension natural order (i.e., lexicographic) to ensure the regressor is
        // constructed correctly.
        for (int i = 0; i < regressedValues.length; i++) {
            regressedValues[mapping[i]] = Linear.predictValues(models.get(i), features, scores);
        }

        Regressor regressor = new Regressor(dimensionNames,regressedValues);
        return new Prediction<>(regressor, features.length - 1, example);
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        int maxFeatures = n < 0 ? featureIDMap.size() : n;
        double[][] featureWeights = getFeatureWeights();

        Comparator<Pair<String, Double>> comparator = Comparator.comparingDouble(p -> Math.abs(p.getB()));
        Map<String, List<Pair<String, Double>>> map = new HashMap<>();
        PriorityQueue<Pair<String, Double>> q = new PriorityQueue<>(maxFeatures, comparator);

        for (int i = 0; i < featureWeights.length; i++) {
            // Exclude bias
            int numFeatures = featureWeights[i].length - 1;
            for (int j = 0; j < numFeatures; j++) {
                Pair<String, Double> cur = new Pair<>(featureIDMap.get(j).getName(), featureWeights[i][j]);
                if (maxFeatures < 0 || q.size() < maxFeatures) {
                    q.offer(cur);
                } else if (comparator.compare(cur, q.peek()) > 0) {
                    q.poll();
                    q.offer(cur);
                }
            }
            List<Pair<String, Double>> list = new ArrayList<>();
            while (q.size() > 0) {
                list.add(q.poll());
            }
            Collections.reverse(list);
            map.put(dimensionNames[mapping[i]], list);
        }

        return map;
    }

    @Override
    protected LibLinearRegressionModel copy(String newName, ModelProvenance newProvenance) {
        List<de.bwaldvogel.liblinear.Model> newModels = new ArrayList<>();
        for (de.bwaldvogel.liblinear.Model m : models) {
            newModels.add(copyModel(m));
        }
        return new LibLinearRegressionModel(newName,newProvenance,featureIDMap,outputIDInfo,newModels);
    }

    @Override
    protected double[][] getFeatureWeights() {
        double[][] featureWeights = new double[models.size()][];

        for (int i = 0; i < models.size(); i++) {
            featureWeights[i] = models.get(i).getFeatureWeights();
        }

        return featureWeights;
    }

    /**
     * The call to model.getFeatureWeights in the public methods copies the
     * weights array so this inner method exists to save the copy in getExcuses.
     * <p>
     * If it becomes a problem then we could cache the feature weights in the
     * model.
     *
     * @param e The example.
     * @param allFeatureWeights The feature weights.
     * @return An excuse for this example.
     */
    @Override
    protected Excuse<Regressor> innerGetExcuse(Example<Regressor> e, double[][] allFeatureWeights) {
        Prediction<Regressor> prediction = predict(e);
        Map<String, List<Pair<String, Double>>> weightMap = new HashMap<>();
        for (int i = 0; i < allFeatureWeights.length; i++) {
            List<Pair<String, Double>> scores = new ArrayList<>();
            for (Feature f : e) {
                int id = featureIDMap.getID(f.getName());
                if (id > -1) {
                    double score = allFeatureWeights[i][id] * f.getValue();
                    scores.add(new Pair<>(f.getName(), score));
                }
            }
            scores.sort((o1, o2) -> o2.getB().compareTo(o1.getB()));
            weightMap.put(dimensionNames[mapping[i]], scores);
        }

        return new Excuse<>(e, prediction, weightMap);
    }

    @Override
    public OnnxMl.ModelProto exportONNXModel(String domain, long modelVersion) {
        ONNXContext onnx = new ONNXContext();

        ONNXPlaceholder input = onnx.floatInput(featureIDMap.size());
        ONNXPlaceholder output = onnx.floatOutput(outputIDInfo.size());
        onnx.setName("Regression-LibLinear");

        return ONNXExportable.buildModel(writeONNXGraph(input).assignTo(output).onnxContext(), domain, modelVersion, this);
    }

    @Override
    public ONNXNode writeONNXGraph(ONNXRef<?> input) {
        ONNXContext onnx = input.onnxContext();
        double[][] weights = new double[models.size()][];
        for (int i = 0; i < models.size(); i++) {
            weights[i] = models.get(i).getFeatureWeights();
        }
        int numFeatures = featureIDMap.size();
        int numOutputs = outputIDInfo.size();

        // Add weights
        ONNXInitializer onnxWeights = onnx.floatTensor("liblinear-weights", Arrays.asList(numFeatures, numOutputs), fb -> {
            for (int j = 0; j < numFeatures; j++) {
                for (int i = 0; i < weights.length; i++) {
                    fb.put((float) weights[i][j]);
                }
            }
        });

        //Add biases
        ONNXInitializer onnxBiases = onnx.floatTensor("liblinear-bias", Collections.singletonList(numOutputs), fb -> {
            // Biases are stored last in the weight vector
            for (int i = 0; i < weights.length; i++) {
                fb.put((float) weights[i][numFeatures]);
            }
        });

        return input.apply(ONNXOperators.GEMM, Arrays.asList(onnxWeights, onnxBiases));
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();

        // Add mapping field to 4.0, 4.1 models and rearrange the dimensions.
        if (mapping == null) {
            this.mapping = ((ImmutableRegressionInfo) outputIDInfo).getIDtoNaturalOrderMapping();
            List<de.bwaldvogel.liblinear.Model> newModels = new ArrayList<>(this.models);

            for (int i = 0; i < mapping.length; i++) {
                newModels.set(i,this.models.get(mapping[i]));
            }

            this.models = Collections.unmodifiableList(newModels);
        }
    }
}
