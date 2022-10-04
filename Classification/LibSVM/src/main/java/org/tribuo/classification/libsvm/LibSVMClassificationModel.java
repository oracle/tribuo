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

package org.tribuo.classification.libsvm;

import ai.onnx.proto.OnnxMl;
import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.ONNXExportable;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.libsvm.protos.LibSVMClassificationModelProto;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.LibSVMTrainer;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.util.Util;
import org.tribuo.util.onnx.ONNXContext;
import org.tribuo.util.onnx.ONNXInitializer;
import org.tribuo.util.onnx.ONNXNode;
import org.tribuo.util.onnx.ONNXOperators;
import org.tribuo.util.onnx.ONNXPlaceholder;
import org.tribuo.util.onnx.ONNXRef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;

/**
 * A classification model that uses an underlying LibSVM model to make the
 * predictions.
 * <p>
 * See:
 * <pre>
 * Chang CC, Lin CJ.
 * "LIBSVM: a library for Support Vector Machines"
 * ACM transactions on intelligent systems and technology (TIST), 2011.
 * </pre>
 * for the nu-svc algorithm:
 * <pre>
 * Schölkopf B, Smola A, Williamson R, Bartlett P L.
 * "New support vector algorithms"
 * Neural Computation, 2000, 1207-1245.
 * </pre>
 * and for the original algorithm:
 * <pre>
 * Cortes C, Vapnik V.
 * "Support-Vector Networks"
 * Machine Learning, 1995.
 * </pre>
 */
public class LibSVMClassificationModel extends LibSVMModel<Label> implements ONNXExportable {
    private static final long serialVersionUID = 3L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * This is used when the model hasn't seen as many outputs as the OutputInfo says are there.
     * It stores the unseen labels to ensure the predict method has the right number of outputs.
     * If there are no unobserved labels it's set to Collections.emptySet.
     */
    private final Set<Label> unobservedLabels;

    LibSVMClassificationModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> labelIDMap, List<svm_model> models) {
        super(name, description, featureIDMap, labelIDMap, models.get(0).param.probability == 1, models);
        // This sets up the unobservedLabels variable.
        int[] curLabels = models.get(0).label;
        if (curLabels.length != labelIDMap.size()) {
            Map<Integer,Label> tmp = new HashMap<>();
            for (Pair<Integer,Label> p : labelIDMap) {
                tmp.put(p.getA(),p.getB());
            }
            for (int i = 0; i < curLabels.length; i++) {
                tmp.remove(i);
            }
            Set<Label> tmpSet = new HashSet<>(tmp.values().size());
            for (Label l : tmp.values()) {
                tmpSet.add(new Label(l.getLabel(),0.0));
            }
            this.unobservedLabels = Collections.unmodifiableSet(tmpSet);
        } else {
            this.unobservedLabels = Collections.emptySet();
        }
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static LibSVMClassificationModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        LibSVMClassificationModelProto proto = message.unpack(LibSVMClassificationModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Label.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a label domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Label> outputDomain = (ImmutableOutputInfo<Label>) carrier.outputDomain();

        svm_model model = deserializeModel(proto.getModel());

        return new LibSVMClassificationModel(carrier.name(),carrier.provenance(),carrier.featureDomain(),outputDomain,Collections.singletonList(model));
    }

    /**
     * Returns the number of support vectors.
     * @return The number of support vectors.
     */
    public int getNumberOfSupportVectors() {
        return models.get(0).SV.length;
    }

    @Override
    public Prediction<Label> predict(Example<Label> example) {
        svm_model model = models.get(0);
        svm_node[] features = LibSVMTrainer.exampleToNodes(example, featureIDMap, null);
        // Bias feature is always set
        if (features.length == 0) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }
        int[] labels = model.label;
        double[] scores = new double[labels.length];
        if (generatesProbabilities) {
            svm.svm_predict_probability(model, features, scores);
        } else {
            //LibSVM returns a one vs one result, and unpacks it into a score vector by voting
            double[] onevone = new double[labels.length * (labels.length - 1) / 2];
            svm.svm_predict_values(model, features, onevone);
            int counter = 0;
            for (int i = 0; i < labels.length; i++) {
                for (int j = i+1; j < labels.length; j++) {
                    if (onevone[counter] > 0) {
                        scores[i]++;
                    } else {
                        scores[j]++;
                    }
                    counter++;
                }
            }
        }
        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        Map<String,Label> map = new LinkedHashMap<>();
        for (int i = 0; i < scores.length; i++) {
            String name = outputIDInfo.getOutput(labels[i]).getLabel();
            Label label = new Label(name, scores[i]);
            map.put(name,label);
            if (label.getScore() > maxScore) {
                maxScore = label.getScore();
                maxLabel = label;
            }
        }
        if (!unobservedLabels.isEmpty()) {
            for (Label l : unobservedLabels) {
                map.put(l.getLabel(),l);
            }
        }
        return new Prediction<>(maxLabel, map, features.length, example, generatesProbabilities);
    }

    @Override
    protected LibSVMClassificationModel copy(String newName, ModelProvenance newProvenance) {
        return new LibSVMClassificationModel(newName,newProvenance,featureIDMap,outputIDInfo,Collections.singletonList(LibSVMModel.copyModel(models.get(0))));
    }

    @Override
    public OnnxMl.ModelProto exportONNXModel(String domain, long modelVersion) {
        ONNXContext onnx = new ONNXContext();

        ONNXPlaceholder input = onnx.floatInput(featureIDMap.size());
        ONNXPlaceholder output = onnx.floatOutput(outputIDInfo.size());
        onnx.setName("Classification-LibSVM");

        writeONNXGraph(input).assignTo(output);
        return ONNXExportable.buildModel(onnx, domain, modelVersion, this);
    }

    @Override
    public ONNXNode writeONNXGraph(ONNXRef<?> input) {
        ONNXContext onnx = input.onnxContext();
        svm_model model = models.get(0);
        int numOneVOne = model.label.length * (model.label.length - 1) / 2;
        int numFeatures = featureIDMap.size();

        // Extract the attributes
        Map<String,Object> attributes = new HashMap<>();
        attributes.put("classlabels_ints",model.label);
        float[] coefficients = new float[model.l * (model.nr_class - 1)];
        for (int i = 0; i < model.nr_class - 1; i++) {
            for (int j = 0; j < model.l; j++) {
                coefficients[i*model.l + j] = (float) model.sv_coef[i][j];
            }
        }
        attributes.put("coefficients",coefficients);
        attributes.put("kernel_params",new float[]{(float)model.param.gamma,(float)model.param.coef0,model.param.degree});
        attributes.put("kernel_type", KernelType.getKernelType(model.param.kernel_type).name());
        float[] rho = new float[model.rho.length];
        for (int i = 0; i < rho.length; i++) {
            rho[i] = (float)-model.rho[i];
        }
        attributes.put("rho",rho);
        // Extract the support vectors
        float[] supportVectors = new float[model.l*numFeatures];

        for (int j = 0; j < model.l; j++) {
            svm_node[] sv = model.SV[j];
            for (svm_node svm_node : sv) {
                int idx = (j * numFeatures) + svm_node.index;
                supportVectors[idx] = (float) svm_node.value;
            }
        }
        attributes.put("support_vectors", supportVectors);
        attributes.put("vectors_per_class", Arrays.copyOf(model.nSV,model.label.length));
        if (generatesProbabilities) {
            attributes.put("prob_a",Arrays.copyOf(Util.toFloatArray(model.probA),numOneVOne));
            attributes.put("prob_b",Arrays.copyOf(Util.toFloatArray(model.probB),numOneVOne));
        }

        // Build SVM node
        List<ONNXNode> outputs = input.apply(ONNXOperators.SVM_CLASSIFIER, Arrays.asList("pred_label", "svm_output"), attributes);
        ONNXNode predLabel = outputs.get(0);
        ONNXNode svmOutput = outputs.get(1);

        ONNXNode ungatheredOutput = svmOutput;
        // if the model is not probabilistic we need to vote the one v one classifier output
        if(!generatesProbabilities) {
            // If the model has two classes then the scores are inverted for some reason
            // This is based on the ONNX Runtime behaviour, but the ONNX SVMClassifier spec is ill-defined
            if(model.nr_class == 2) {
                ONNXInitializer negOne = onnx.constant("minus_one", -1.0f);
                ungatheredOutput = writeDecisionFunction(svmOutput.apply(ONNXOperators.MUL, negOne));
            } else {
                ungatheredOutput = writeDecisionFunction(svmOutput);
            }
        }

        // Undo the libsvm mapping so the indices line up with Tribuo indices
        int[] backwardsLibSVMMapping = new int[model.label.length];
        for (int i = 0; i < model.label.length; i++) {
            backwardsLibSVMMapping[model.label[i]] = i;
        }

        ONNXInitializer indices = onnx.array("label_indices", backwardsLibSVMMapping);

        return ungatheredOutput.apply(ONNXOperators.GATHER, indices, Collections.singletonMap("axis", 1));
    }

    private ONNXNode writeDecisionFunction(ONNXNode svmOutputName) {
        final ONNXContext onnx = svmOutputName.onnxContext();
        ONNXInitializer one = onnx.constant("one", 1.0f);
        ONNXInitializer zero = onnx.constant("zero", 0.0f);

        ONNXNode prediction = svmOutputName.apply(ONNXOperators.LESS, zero).cast(float.class);

        svm_model model = models.get(0);

        TreeMap<Integer, List<ONNXNode>> votes = new TreeMap<>();

        int k = 0;
        for (int i = 0; i < model.nr_class; i++) {
            for (int j = i + 1; j < model.nr_class; j++) {
                ONNXInitializer index = onnx.constant("Vind_" + k, (long) k);

                ONNXNode extractedFeature = prediction.apply(ONNXOperators.ARRAY_FEATURE_EXTRACTOR, index, "Vsvcv_" + k);
                votes.computeIfAbsent(j, x -> new ArrayList<>()).add(extractedFeature);

                ONNXNode addNeg = extractedFeature.apply(ONNXOperators.NEG, "Vnegv_" + k).apply(ONNXOperators.ADD, one, "Vnegv1_" + k);
                votes.computeIfAbsent(i, x -> new ArrayList<>()).add(addNeg);

                k += 1;
            }
        }

        List<ONNXNode> oneVOneVotes = votes.values().stream()
                .map(nodes -> onnx.operation(ONNXOperators.SUM, nodes, "svm_votes"))
                .collect(Collectors.toList());
                /*
                votes.entrySet().stream().sequential()
                .sorted(Comparator.comparingInt(Map.Entry::getKey))
                .map(Map.Entry::getValue)
                .map(nodes -> onnx.operation(ONNXOperators.SUM, nodes, "svm_votes"))
                .collect(Collectors.toList());

                 */

        return onnx.operation(ONNXOperators.CONCAT, oneVOneVotes, "svm_output", Collections.singletonMap("axis", 1));
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<Label> carrier = createDataCarrier();

        LibSVMClassificationModelProto.Builder modelBuilder = LibSVMClassificationModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setModel(serializeModel(models.get(0)));

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(LibSVMClassificationModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }
}
