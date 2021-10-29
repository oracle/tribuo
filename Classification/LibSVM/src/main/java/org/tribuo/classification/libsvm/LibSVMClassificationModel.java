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

package org.tribuo.classification.libsvm;

import ai.onnx.proto.OnnxMl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.Tribuo;
import org.tribuo.classification.Label;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.LibSVMTrainer;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXExportable;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.onnx.ONNXShape;
import org.tribuo.onnx.ONNXUtils;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.util.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
        ONNXContext context = new ONNXContext();

        // Build graph
        OnnxMl.GraphProto graph = exportONNXGraph(context);

        // Build model
        OnnxMl.ModelProto.Builder builder = OnnxMl.ModelProto.newBuilder();
        builder.setGraph(graph);
        builder.setDomain(domain);
        builder.setProducerName("Tribuo");
        builder.setProducerVersion(Tribuo.VERSION);
        builder.setModelVersion(modelVersion);
        builder.setDocString(toString());
        builder.addOpsetImport(ONNXOperators.getOpsetProto());
        builder.setIrVersion(6);

        // Extract provenance and store in metadata
        OnnxMl.StringStringEntryProto.Builder metaBuilder = OnnxMl.StringStringEntryProto.newBuilder();
        metaBuilder.setKey(ONNXExportable.PROVENANCE_METADATA_FIELD);
        metaBuilder.setValue(serializeProvenance(getProvenance()));
        builder.addMetadataProps(metaBuilder.build());

        return builder.build();
    }

    @Override
    public OnnxMl.GraphProto exportONNXGraph(ONNXContext context) {
        OnnxMl.GraphProto.Builder graphBuilder = OnnxMl.GraphProto.newBuilder();
        graphBuilder.setName("LibSVM-Classification");

        svm_model model = models.get(0);
        int numOneVOne = model.label.length * (model.label.length - 1) / 2;
        int numFeatures = featureIDMap.size();

        // Make inputs and outputs
        OnnxMl.TypeProto inputType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,featureIDMap.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto inputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(inputType).setName("input").build();
        graphBuilder.addInput(inputValueProto);
        OnnxMl.TypeProto outputScoresType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,outputIDInfo.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto outputScoresProto = OnnxMl.ValueInfoProto.newBuilder().setType(outputScoresType).setName("output").build();
        graphBuilder.addOutput(outputScoresProto);

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
        String[] outputs = new String[]{"pred_label", "svm_output"};
        OnnxMl.NodeProto svm = ONNXOperators.SVM_CLASSIFIER.build(context,inputValueProto.getName(),outputs,attributes);
        graphBuilder.addNode(svm);

        String outputName = "svm_output";
        if (model.nr_class == 2) {
            OnnxMl.TensorProto negOne = ONNXUtils.scalarBuilder(context, "minus_one", -1.0f);
            graphBuilder.addInitializer(negOne);
            OnnxMl.NodeProto binaryOutput = ONNXOperators.MUL.build(context, new String[]{outputs[1], negOne.getName()}, "svm_output_b");
            graphBuilder.addNode(binaryOutput);
            outputName = "svm_output_b";
        }

        if (!generatesProbabilities) {
            writeDecisionFunction(context, graphBuilder, outputs[1], "ungathered_output");
            outputName = "ungathered_output";
        }

        int[] backwardsLibSVMMapping = new int[model.label.length];
        for (int i = 0; i < model.label.length; i++) {
            backwardsLibSVMMapping[model.label[i]] = i;
        }

        OnnxMl.TensorProto indices = ONNXUtils.arrayBuilder(context, "label_indices", backwardsLibSVMMapping);
        graphBuilder.addInitializer(indices);

        OnnxMl.NodeProto gather = ONNXOperators.GATHER.build(context, new String[]{outputName, indices.getName()}, "output", Collections.singletonMap("axis", 1));
        graphBuilder.addNode(gather);

        return graphBuilder.build();
    }

    private void writeDecisionFunction(ONNXContext context, OnnxMl.GraphProto.Builder builder, String svmOutputName, String outputName) {
        // cst1
        OnnxMl.TensorProto one = ONNXUtils.scalarBuilder(context,"one",1.0f);
        builder.addInitializer(one);
        // cst0
        OnnxMl.TensorProto zero = ONNXUtils.scalarBuilder(context,"zero",0.0f);
        builder.addInitializer(zero);

        OnnxMl.NodeProto prediction = ONNXOperators.LESS.build(context,new String[]{svmOutputName,zero.getName()},context.generateUniqueName("prediction"));
        builder.addNode(prediction);

        OnnxMl.NodeProto floatPrediction = ONNXOperators.CAST.build(context,prediction.getOutput(0),
                context.generateUniqueName("float_prediction"),
                Collections.singletonMap("to",OnnxMl.TensorProto.DataType.FLOAT.getNumber()));
        builder.addNode(floatPrediction);

        svm_model model = models.get(0);
        String[] oneVOneNames = new String[model.nr_class];
        LinkedHashMap<String,List<OnnxMl.NodeProto>> voteAdds = new LinkedHashMap<>();
        for (int i = 0; i < model.nr_class; i++) {
            oneVOneNames[i] = context.generateUniqueName("svcvote_" + i);
            voteAdds.put(oneVOneNames[i],new ArrayList<>());
        }

        int k = 0;
        for (int i = 0; i < model.nr_class; i++) {
            for (int j = i+1; j < model.nr_class; j++) {
                OnnxMl.TensorProto index = ONNXUtils.scalarBuilder(context,"Vind_" + k, (long) k);
                builder.addInitializer(index);
                OnnxMl.NodeProto featureExtractor = ONNXOperators.ARRAY_FEATURE_EXTRACTOR.build(context,
                        new String[]{floatPrediction.getOutput(0),index.getName()},
                        context.generateUniqueName("Vsvcv_"+k)
                );
                builder.addNode(featureExtractor);
                voteAdds.get(oneVOneNames[j]).add(featureExtractor);
                OnnxMl.NodeProto negate = ONNXOperators.NEG.build(context,featureExtractor.getOutput(0),context.generateUniqueName("Vnegv_"+k));
                builder.addNode(negate);
                OnnxMl.NodeProto addNeg = ONNXOperators.ADD.build(context,new String[]{negate.getOutput(0),one.getName()},context.generateUniqueName("Vnegv1_"+k));
                builder.addNode(addNeg);
                voteAdds.get(oneVOneNames[i]).add(addNeg);

                k += 1;
            }
        }

        for (Map.Entry<String, List<OnnxMl.NodeProto>> e : voteAdds.entrySet()) {
            String[] nodeNames = e.getValue().stream().map((OnnxMl.NodeProto n) -> n.getOutput(0)).toArray(String[]::new);
            OnnxMl.NodeProto sum = ONNXOperators.SUM.build(context,nodeNames,e.getKey());
            builder.addNode(sum);
        }

        OnnxMl.NodeProto concat = ONNXOperators.CONCAT.build(context,oneVOneNames,outputName,Collections.singletonMap("axis",1));
        builder.addNode(concat);
    }

}
