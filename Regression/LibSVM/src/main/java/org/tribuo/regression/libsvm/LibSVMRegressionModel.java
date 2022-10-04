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

package org.tribuo.regression.libsvm;

import ai.onnx.proto.OnnxMl;
import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.ONNXExportable;
import org.tribuo.Prediction;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.LibSVMTrainer;
import org.tribuo.common.libsvm.protos.SVMModelProto;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.ImmutableRegressionInfo;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.libsvm.protos.LibSVMRegressionModelProto;
import org.tribuo.util.Util;
import org.tribuo.util.onnx.ONNXContext;
import org.tribuo.util.onnx.ONNXInitializer;
import org.tribuo.util.onnx.ONNXNode;
import org.tribuo.util.onnx.ONNXOperators;
import org.tribuo.util.onnx.ONNXPlaceholder;
import org.tribuo.util.onnx.ONNXRef;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

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
     * Deserialization constructor.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The features this model knows about.
     * @param outputIDInfo The outputs this model can produce.
     * @param models The svm_models themselves.
     * @param means The output dimension means.
     * @param variances The output dimension variances.
     * @param standardized Is the model standardized?
     */
    private LibSVMRegressionModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo, List<svm_model> models, double[] means, double[] variances, boolean standardized) {
        super(name, provenance, featureIDMap, outputIDInfo, false, models);
        this.dimensionNames = Regressor.extractNames(outputIDInfo);
        this.means = means;
        this.variances = variances;
        this.standardized = standardized;
        this.mapping = ((ImmutableRegressionInfo) outputIDInfo).getIDtoNaturalOrderMapping();
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static LibSVMRegressionModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        LibSVMRegressionModelProto proto = message.unpack(LibSVMRegressionModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Regressor.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a regression domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Regressor> outputDomain = (ImmutableOutputInfo<Regressor>) carrier.outputDomain();

        if (outputDomain.size() != proto.getModelCount()) {
            throw new IllegalStateException("Invalid protobuf, did not find a model for each output dimension, expected " + outputDomain.size() + " found " + proto.getModelCount());
        }
        List<svm_model> models = new ArrayList<>();
        for (SVMModelProto modelProto : proto.getModelList()) {
            models.add(deserializeModel(modelProto));
        }
        double[] means = proto.getMeansCount() == 0 ? null : Util.toPrimitiveDouble(proto.getMeansList());
        if (means != null && means.length != outputDomain.size()) {
            throw new IllegalStateException("Invalid protobuf, expected " + outputDomain.size() + " means, found " + means.length);
        }
        double[] variances = proto.getVariancesCount() == 0 ? null : Util.toPrimitiveDouble(proto.getVariancesList());
        if (variances != null && variances.length != outputDomain.size()) {
            throw new IllegalStateException("Invalid protobuf, expected " + outputDomain.size() + " variances, found " + variances.length);
        }

        return new LibSVMRegressionModel(carrier.name(),carrier.provenance(),carrier.featureDomain(),outputDomain,Collections.unmodifiableList(models),means,variances,proto.getStandardized());
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
        ONNXContext onnx = new ONNXContext();

        ONNXPlaceholder input = onnx.floatInput(featureIDMap.size());
        ONNXPlaceholder output = onnx.floatOutput(outputIDInfo.size());
        onnx.setName("Regression-LibSVM");

        return ONNXExportable.buildModel(writeONNXGraph(input).assignTo(output).onnxContext(), domain, modelVersion, this);
    }

    private static ONNXNode buildONNXSVMRegressor(int numFeatures, ONNXRef<?> input, svm_model model) {
        // Extract the attributes
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("coefficients", Util.toFloatArray(model.sv_coef[0]));
        attributes.put("kernel_params", new float[]{(float) model.param.gamma, (float) model.param.coef0, model.param.degree});
        attributes.put("kernel_type", KernelType.getKernelType(model.param.kernel_type).name());
        attributes.put("n_supports", model.l);
        attributes.put("one_class", 0);
        attributes.put("rho", new float[]{(float) -model.rho[0]});
        // Extract the support vectors
        float[] supportVectors = new float[model.l * numFeatures];

        for (int j = 0; j < model.l; j++) {
            svm_node[] sv = model.SV[j];
            for (svm_node svm_node : sv) {
                int idx = (j * numFeatures) + svm_node.index;
                supportVectors[idx] = (float) svm_node.value;
            }
        }
        attributes.put("support_vectors", supportVectors);
        return input.apply(ONNXOperators.SVM_REGRESSOR, attributes);
    }

    @Override
    public ONNXNode writeONNXGraph(ONNXRef<?> input) {
        ONNXContext onnx = input.onnxContext();

        final int numFeatures = featureIDMap.size();


        // Make the individual SVM Regressors for each dimension
        List<ONNXNode> outputs = models.stream()
                .map(model -> buildONNXSVMRegressor(numFeatures, input, model))
                .collect(Collectors.toList());

        // Make concat to bring them all together
        ONNXNode concat = onnx.operation(ONNXOperators.CONCAT, outputs, "concat_output", Collections.singletonMap("axis", 1));

        if(standardized) {
            ONNXInitializer outputMean = onnx.array("y_mean", means);
            ONNXInitializer outputVariance = onnx.array("y_variances", variances);

            return concat.apply(ONNXOperators.MUL, outputVariance)
                    .apply(ONNXOperators.ADD, outputMean);
        } else {
            return concat;
        }
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<Regressor> carrier = createDataCarrier();

        LibSVMRegressionModelProto.Builder modelBuilder = LibSVMRegressionModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        for (svm_model m : models) {
            modelBuilder.addModel(serializeModel(m));
        }
        if (means != null) {
            modelBuilder.addAllMeans(Arrays.stream(means).boxed().collect(Collectors.toList()));
        }
        if (variances != null) {
            modelBuilder.addAllVariances(Arrays.stream(variances).boxed().collect(Collectors.toList()));
        }
        modelBuilder.setStandardized(standardized);

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(LibSVMRegressionModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
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
