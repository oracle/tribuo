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

package org.tribuo.regression.libsvm;

import ai.onnx.proto.OnnxMl;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.Tribuo;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.LibSVMTrainer;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXExportable;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.onnx.ONNXShape;
import org.tribuo.onnx.ONNXUtils;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.ImmutableRegressionInfo;
import org.tribuo.regression.Regressor;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import org.tribuo.util.Util;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A regression model that uses an underlying libSVM model to make the
 * predictions. Contains an independent model for each output dimension.
 * <p>
 * See:
 * <pre>
 * Chang CC, Lin CJ.
 * "LIBSVM: a library for Support Vector Machines"
 * ACM transactions on intelligent systems and technology (TIST), 2011.
 * </pre>
 * for the nu-svr algorithm:
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
public class LibSVMRegressionModel extends LibSVMModel<Regressor> implements ONNXExportable {
    private static final long serialVersionUID = 2L;

    private final String[] dimensionNames;

    private double[] means;

    private double[] variances;

    private final boolean standardized;

    // Not final as it doesn't exist in 4.0 or 4.1 and so must be created on deserialization.
    private int[] mapping;

    /**
     * Constructs a LibSVMRegressionModel with regular outputs.
     * @param name The model name.
     * @param description The model provenance.
     * @param featureIDMap The features this model knows about.
     * @param outputIDInfo The outputs this model can produce.
     * @param models The svm_models themselves.
     */
    LibSVMRegressionModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo, List<svm_model> models) {
        super(name, description, featureIDMap, outputIDInfo, false, models);
        this.dimensionNames = Regressor.extractNames(outputIDInfo);
        this.means = null;
        this.variances = null;
        this.standardized = false;
        this.mapping = ((ImmutableRegressionInfo) outputIDInfo).getIDtoNaturalOrderMapping();
    }

    /**
     * Constructs a LibSVMRegressionModel with standardized outputs that must be upscaled during prediction.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The features this model knows about.
     * @param outputIDInfo The outputs this model can produce.
     * @param models The svm_models themselves.
     * @param means The output dimension means.
     * @param variances The output dimension variances.
     */
    LibSVMRegressionModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo, List<svm_model> models, double[] means, double[] variances) {
        super(name, provenance, featureIDMap, outputIDInfo, false, models);
        this.dimensionNames = Regressor.extractNames(outputIDInfo);
        this.means = means;
        this.variances = variances;
        this.standardized = true;
        this.mapping = ((ImmutableRegressionInfo) outputIDInfo).getIDtoNaturalOrderMapping();
    }

    /**
     * Is this LibSVMRegressionModel operating in a standardized space.
     * @return True if the model has been standardized.
     */
    boolean isStandardized() {
        return standardized;
    }

    /**
     * Returns the support vectors used for each dimension.
     * @return The support vectors.
     */
    public Map<String,Integer> getNumberOfSupportVectors() {
        Map<String,Integer> output = new HashMap<>();

        for (int i = 0; i < dimensionNames.length; i++) {
            output.put(dimensionNames[i],models.get(i).SV.length);
        }

        return output;
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        svm_node[] features = LibSVMTrainer.exampleToNodes(example, featureIDMap, null);
        if (features.length == 0) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }
        double[] scores = new double[1];
        double[] regressedValues = new double[models.size()];

        for (int i = 0; i < regressedValues.length; i++) {
            regressedValues[mapping[i]] = svm.svm_predict_values(models.get(i), features, scores);
            if (standardized) {
                regressedValues[i] = (regressedValues[i] * variances[mapping[i]]) + means[mapping[i]];
            }
        }

        Regressor regressor = new Regressor(dimensionNames,regressedValues);
        return new Prediction<>(regressor, features.length, example);
    }

    /**
     * Accessor used in the tests. May be considered to be made public.
     * @return The means used for standardization.
     */
    protected double[] getMeans() {
        if (means != null) {
            return Arrays.copyOf(means, means.length);
        } else {
            return null;
        }
    }

    /**
     * Accessor used in the tests. May be considered to be made public.
     * @return The variances used for standardization.
     */
    protected double[] getVariances() {
        if (variances != null) {
            return Arrays.copyOf(variances, variances.length);
        } else {
            return null;
        }
    }

    @Override
    protected LibSVMRegressionModel copy(String newName, ModelProvenance newProvenance) {
        List<svm_model> newModels = new ArrayList<>();
        for (svm_model m : models) {
            newModels.add(copyModel(m));
        }
        return new LibSVMRegressionModel(newName,newProvenance,featureIDMap,outputIDInfo,newModels);
    }

    @Override
    public OnnxMl.ModelProto exportONNXModel(String domain, long modelVersion) {
        ONNXContext context = new ONNXContext();

        // Make inputs and outputs
        OnnxMl.TypeProto inputType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,featureIDMap.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto inputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(inputType).setName("input").build();
        context.addInput(inputValueProto);
        OnnxMl.TypeProto outputType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,outputIDInfo.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto outputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(outputType).setName("output").build();
        context.addOutput(outputValueProto);
        context.setName("Regression-LibSVM");

        // Build graph
        writeONNXGraph(context, inputValueProto.getName(), outputValueProto.getName());

        // Build model
        OnnxMl.ModelProto.Builder builder = OnnxMl.ModelProto.newBuilder();
        builder.setGraph(context.buildGraph());
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
    public void writeONNXGraph(ONNXContext context, String inputName, String outputName)  {
        int numFeatures = featureIDMap.size();

        // Make the individual SVM Regressors for each dimension
        String[] outputNames = new String[models.size()];
        for (int i = 0; i < models.size(); i++) {
            svm_model model = models.get(i);
            // Extract the attributes
            Map<String, Object> attributes = new HashMap<>();
            attributes.put("coefficients", Util.toFloatArray(model.sv_coef[0]));
            attributes.put("kernel_params", new float[]{(float) model.param.gamma, (float) model.param.coef0, model.param.degree});
            attributes.put("kernel_type", KernelType.getKernelType(model.param.kernel_type).name());
            attributes.put("n_supports", model.l);
            attributes.put("one_class", 0);
            attributes.put("rho", new float[]{(float)-model.rho[0]});
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

            // Build SVM node
            outputNames[i] = context.generateUniqueName("dimension_output");
            OnnxMl.NodeProto svm = ONNXOperators.SVM_REGRESSOR.build(context, inputName, outputNames[i], attributes);
            context.addNode(svm);
        }

        String concatName = standardized ? context.generateUniqueName("concat_output") : outputName;
        // Make concat to bring them all together
        OnnxMl.NodeProto concat = ONNXOperators.CONCAT.build(context, outputNames, concatName, Collections.singletonMap("axis", 1));
        context.addNode(concat);

        if (standardized) {
            // Add output means
            OnnxMl.TensorProto outputMeanProto = ONNXUtils.arrayBuilder(context,"y_mean",means);
            context.addInitializer(outputMeanProto);

            // Add output variances
            OnnxMl.TensorProto outputVarianceProto = ONNXUtils.arrayBuilder(context, "y_var",variances);
            context.addInitializer(outputVarianceProto);

            // Scale outputs
            String varianceOutput = context.generateUniqueName("y_var_scale_output");
            OnnxMl.NodeProto varianceScale = ONNXOperators.MUL.build(context, new String[]{concatName,outputVarianceProto.getName()}, varianceOutput);
            context.addNode(varianceScale);

            OnnxMl.NodeProto meanScale = ONNXOperators.ADD.build(context, new String[]{varianceOutput,outputMeanProto.getName()}, outputName);
            context.addNode(meanScale);
        }
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();

        // Add mapping field to 4.0, 4.1 models and rearrange the dimensions.
        if (mapping == null) {
            this.mapping = ((ImmutableRegressionInfo) outputIDInfo).getIDtoNaturalOrderMapping();
            List<svm_model> newModels = new ArrayList<>(this.models);
            double[] newMeans = new double[newModels.size()];
            double[] newVariances = new double[newModels.size()];

            for (int i = 0; i < mapping.length; i++) {
                newModels.set(i,this.models.get(mapping[i]));
                if (this.means != null) {
                    newMeans[i] = this.means[mapping[i]];
                    newVariances[i] = this.variances[mapping[i]];
                }
            }

            this.models = Collections.unmodifiableList(newModels);
            this.means = newMeans;
            this.variances = newVariances;
        }
    }
}
