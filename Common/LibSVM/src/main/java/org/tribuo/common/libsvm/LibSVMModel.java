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

package org.tribuo.common.libsvm;

import com.oracle.labs.mlrg.olcut.util.Pair;
import libsvm.svm_parameter;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.common.libsvm.protos.SVMModelProto;
import org.tribuo.common.libsvm.protos.SVMNodeArrayProto;
import org.tribuo.common.libsvm.protos.SVMParameterProto;
import org.tribuo.provenance.ModelProvenance;
import libsvm.svm_model;
import libsvm.svm_node;
import org.tribuo.util.Util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * A model that uses an underlying libSVM model to make the
 * predictions.
 * <p>
 * See:
 * <pre>
 * Chang CC, Lin CJ.
 * "LIBSVM: a library for Support Vector Machines"
 * ACM transactions on intelligent systems and technology (TIST), 2011.
 * </pre>
 * for the nu-svm algorithm:
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
public abstract class LibSVMModel<T extends Output<T>> extends Model<T> implements Serializable {
    private static final long serialVersionUID = 3L;

    private static final Logger logger = Logger.getLogger(LibSVMModel.class.getName());

    /**
     * The LibSVM models. Multiple models are used for multi-label or multidimensional regression outputs.
     * <p>
     * Not final to support deserialization reordering of multidimensional regression models which have an incorrect id mapping.
     * Will be final again in some future version which doesn't maintain serialization compatibility with 4.X.
     */
    protected List<svm_model> models;

    /**
     * Constructs a LibSVMModel from the supplied arguments.
     * @param name The model name.
     * @param description The model provenance.
     * @param featureIDMap The features the model knows about.
     * @param outputIDInfo The outputs the model can produce.
     * @param generatesProbabilities Does the model generate probabilities or not?
     * @param models The svm models themselves.
     */
    protected LibSVMModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, boolean generatesProbabilities, List<svm_model> models) {
        super(name, description, featureIDMap, outputIDInfo, generatesProbabilities);
        this.models = models;
    }

    /**
     * Returns an unmodifiable copy of the underlying list of libsvm models.
     * @deprecated Deprecated to unify the names across LibLinear, LibSVM and XGBoost.
     * @return The underlying model list.
     */
    @Deprecated
    public List<svm_model> getModel() {
        return getInnerModels();
    }

    /**
     * Returns an unmodifiable copy of the underlying list of libsvm models.
     * @return The underlying model list.
     */
    public List<svm_model> getInnerModels() {
        List<svm_model> copy = new ArrayList<>();

        for (svm_model m : models) {
            copy.add(copyModel(m));
        }

        return Collections.unmodifiableList(copy);
    }

    @Override
    public Map<String, List<Pair<String,Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        return Optional.empty();
    }

    /**
     * Copies an svm_model, as it does not provide a copy method.
     *
     * @param model The svm_model to copy.
     * @return A deep copy of the model.
     */
    protected static svm_model copyModel(svm_model model) {
        svm_model newModel = new svm_model();

        newModel.param = SVMParameters.copyParameters(model.param);
        newModel.l = model.l;
        newModel.nr_class = model.nr_class;
        newModel.rho = model.rho != null ? Arrays.copyOf(model.rho,model.rho.length) : null;
        newModel.probA = model.probA != null ? Arrays.copyOf(model.probA,model.probA.length) : null;
        newModel.probB = model.probB != null ? Arrays.copyOf(model.probB,model.probB.length) : null;
        newModel.label = model.label != null ? Arrays.copyOf(model.label,model.label.length) : null;
        newModel.sv_indices = model.sv_indices != null ? Arrays.copyOf(model.sv_indices,model.sv_indices.length) : null;
        newModel.nSV = model.nSV != null ? Arrays.copyOf(model.nSV,model.nSV.length) : null;
        if (model.SV != null) {
            newModel.SV = new svm_node[model.SV.length][];
            for (int i = 0; i < newModel.SV.length; i++) {
                if (model.SV[i] != null) {
                    svm_node[] copy = new svm_node[model.SV[i].length];
                    for (int j = 0; j < copy.length; j++) {
                        if (model.SV[i][j] != null) {
                            svm_node curCopy = new svm_node();
                            curCopy.index = model.SV[i][j].index;
                            curCopy.value = model.SV[i][j].value;
                            copy[j] = curCopy;
                        }
                    }
                    newModel.SV[i] = copy;
                }
            }
        }
        if (model.sv_coef != null) {
            newModel.sv_coef = new double[model.sv_coef.length][];
            for (int i = 0; i < newModel.sv_coef.length; i++) {
                if (model.sv_coef[i] != null) {
                    newModel.sv_coef[i] = Arrays.copyOf(model.sv_coef[i],model.sv_coef[i].length);
                }
            }
        }

        return newModel;
    }

    /**
     * Checks for equality between two svm_models.
     * <p>
     * Equality is defined as bit-wise exact for SV, rho, sv_coeff, probA and probB.
     * @param first The first model.
     * @param second The second model.
     * @return True if the models are identical.
     */
    public static boolean modelEquals(svm_model first, svm_model second) {
        boolean svCoeffEquals = Arrays.deepEquals(first.sv_coef, second.sv_coef);
        boolean probAEquals = Arrays.equals(first.probA, second.probA);
        boolean probBEquals = Arrays.equals(first.probB, second.probB);
        boolean nSVEquals = Arrays.equals(first.nSV, second.nSV);
        boolean rhoEquals = Arrays.equals(first.rho, second.rho);
        boolean labelEquals = Arrays.equals(first.label, second.label);
        if (svCoeffEquals && probAEquals && probBEquals && nSVEquals && rhoEquals && labelEquals) {
            // Check SVs.
            try {
                for (int i = 0; i < first.SV.length; i++) {
                    for (int j = 0; j < first.SV[i].length; j++) {
                        svm_node firstNode = first.SV[i][j];
                        svm_node secondNode = second.SV[i][j];
                        if (firstNode.index != secondNode.index) {
                            return false;
                        } else if (Double.compare(firstNode.value,secondNode.value) != 0) {
                            return false;
                        }
                    }
                }
                return true;
            } catch (NullPointerException e) {
                return false;
            }
        } else {
            return false;
        }
    }

    /**
     * Serializes a svm_parameter to a protobuf.
     * @param param The parameters to serialize.
     * @return The protobuf.
     */
    protected static SVMParameterProto serializeParameters(svm_parameter param) {
        SVMParameterProto.Builder paramBuilder = SVMParameterProto.newBuilder();
        paramBuilder.setSvmType(param.svm_type);
        paramBuilder.setKernelType(param.kernel_type);
        paramBuilder.setDegree(param.degree);
        paramBuilder.setGamma(param.gamma);
        paramBuilder.setCoef0(param.coef0);
        paramBuilder.setCacheSize(param.cache_size);
        paramBuilder.setEps(param.eps);
        paramBuilder.setC(param.C);
        paramBuilder.setNrWeight(param.nr_weight);
        paramBuilder.setNu(param.nu);
        paramBuilder.setP(param.p);
        paramBuilder.setShrinking(param.shrinking);
        paramBuilder.setProbability(param.probability);
        if (param.weight != null) {
            for (int i = 0; i < param.weight.length; i++) {
                paramBuilder.addWeight(param.weight[i]);
            }
        }
        if (param.weight_label != null) {
            for (int i = 0; i < param.weight_label.length; i++) {
                paramBuilder.addWeightLabel(param.weight_label[i]);
            }
        }
        return paramBuilder.build();
    }

    /**
     * Serializes a LibSVM svm_model to a protobuf.
     * @param model The model to serialize.
     * @return The protobuf representation.
     */
    protected static SVMModelProto serializeModel(svm_model model) {
        SVMModelProto.Builder modelBuilder = SVMModelProto.newBuilder();
        modelBuilder.setParam(serializeParameters(model.param));
        modelBuilder.setNrClass(model.nr_class);
        modelBuilder.setL(model.l);
        if (model.SV != null) {
            for (int i = 0; i < model.SV.length; i++) {
                SVMNodeArrayProto.Builder nodeBuilder = SVMNodeArrayProto.newBuilder();
                for (int j = 0; j < model.SV[i].length; j++) {
                    nodeBuilder.addIndex(model.SV[i][j].index);
                    nodeBuilder.addValue(model.SV[i][j].value);
                }
                modelBuilder.addSV(nodeBuilder.build());
            }
        }
        if (model.sv_coef != null) {
            for (int i = 0; i < model.sv_coef.length; i++) {
                modelBuilder.addSvCoefLengths(model.sv_coef[i].length);
                for (int j = 0; j < model.sv_coef[i].length; j++) {
                    modelBuilder.addSvCoef(model.sv_coef[i][j]);
                }
            }
        }
        if (model.rho != null) {
            modelBuilder.addAllRho(Arrays.stream(model.rho).boxed().collect(Collectors.toList()));
        }
        if (model.probA != null) {
            modelBuilder.addAllProbA(Arrays.stream(model.probA).boxed().collect(Collectors.toList()));
        }
        if (model.probB != null) {
            modelBuilder.addAllProbB(Arrays.stream(model.probB).boxed().collect(Collectors.toList()));
        }
        if (model.sv_indices != null) {
            modelBuilder.addAllSvIndices(Arrays.stream(model.sv_indices).boxed().collect(Collectors.toList()));
        }
        if (model.label != null) {
            modelBuilder.addAllLabel(Arrays.stream(model.label).boxed().collect(Collectors.toList()));
        }
        if (model.nSV != null) {
            modelBuilder.addAllNSV(Arrays.stream(model.nSV).boxed().collect(Collectors.toList()));
        }
        return modelBuilder.build();
    }

    /**
     * Deserializes svm_parameter from a protobuf.
     * @param proto The protobuf to deserialize.
     * @return The parameters.
     */
    protected static svm_parameter deserializeParameters(SVMParameterProto proto) {
        svm_parameter param = new svm_parameter();
        param.svm_type = proto.getSvmType();
        param.kernel_type = proto.getKernelType();
        param.degree = proto.getDegree();
        param.gamma = proto.getGamma();
        param.coef0 = proto.getCoef0();
        param.cache_size = proto.getCacheSize();
        param.eps = proto.getEps();
        param.C = proto.getC();
        param.nr_weight = proto.getNrWeight();
        param.nu = proto.getNu();
        param.p = proto.getP();
        param.shrinking = proto.getShrinking();
        param.probability = proto.getProbability();
        if (proto.getWeightCount() > 0) {
            param.weight = new double[proto.getWeightCount()];
            for (int i = 0; i < proto.getWeightCount(); i++) {
                param.weight[i] = proto.getWeight(i);
            }
        }
        if (proto.getWeightLabelCount() > 0) {
            param.weight_label = new int[proto.getWeightLabelCount()];
            for (int i = 0; i < proto.getWeightLabelCount(); i++) {
                param.weight_label[i] = proto.getWeightLabel(i);
            }
        }
        return param;
    }

    /**
     * Deserializes a LibSVM svm_model from a protobuf.
     * @param proto The protobuf to deserialize.
     * @return The svm_model.
     */
    protected static svm_model deserializeModel(SVMModelProto proto) {
        svm_model model = new svm_model();
        model.param = deserializeParameters(proto.getParam());
        model.nr_class = proto.getNrClass();
        model.l = proto.getL();
        if (proto.getSVCount() > 0) {
            model.SV = new svm_node[proto.getSVCount()][];
            for (int i = 0; i < proto.getSVCount(); i++) {
                SVMNodeArrayProto nodes = proto.getSV(i);
                if (nodes.getIndexCount() != nodes.getValueCount()) {
                    throw new IllegalStateException("Invalid protobuf, expected the same number of indices and values, found " + nodes.getIndexCount() + " indices and " + nodes.getValueCount() + " values");
                } else {
                    model.SV[i] = new svm_node[nodes.getIndexCount()];
                    for (int j = 0; j < nodes.getIndexCount(); j++) {
                        model.SV[i][j] = new svm_node();
                        model.SV[i][j].index = nodes.getIndex(j);
                        model.SV[i][j].value = nodes.getValue(j);
                    }
                }
            }
        }
        if (proto.getSvCoefCount() > 0) {
            if (proto.getSvCoefLengthsCount() > 0) {
                model.sv_coef = new double[proto.getSvCoefLengthsCount()][];
                for (int i = 0; i < proto.getSvCoefLengthsCount(); i++) {
                    model.sv_coef[i] = new double[proto.getSvCoefLengths(i)];
                }
                int i = 0;
                int j = 0;
                for (int k = 0; k < proto.getSvCoefCount(); k++) {
                    model.sv_coef[i][j] = proto.getSvCoef(k);
                    j++;
                    if (j >= proto.getSvCoefLengths(i)) {
                        i++;
                        j = 0;
                    }
                }
            } else {
                throw new IllegalStateException("Invalid protobuf, length counts not stored for sv_coef");
            }
        }
        if (proto.getRhoCount() > 0) {
            model.rho = Util.toPrimitiveDouble(proto.getRhoList());
        }
        if (proto.getProbACount() > 0) {
            model.probA = Util.toPrimitiveDouble(proto.getProbAList());
        }
        if (proto.getProbBCount() > 0) {
            model.probB = Util.toPrimitiveDouble(proto.getProbBList());
        }
        if (proto.getSvIndicesCount() > 0) {
            model.sv_indices = Util.toPrimitiveInt(proto.getSvIndicesList());
        }
        if (proto.getLabelCount() > 0) {
            model.label = Util.toPrimitiveInt(proto.getLabelList());
        }
        if (proto.getNSVCount() > 0) {
            model.nSV = Util.toPrimitiveInt(proto.getNSVList());
        }

        return model;
    }
}
