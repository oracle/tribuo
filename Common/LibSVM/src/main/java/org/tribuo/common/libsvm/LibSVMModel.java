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
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.provenance.ModelProvenance;
import libsvm.svm_model;
import libsvm.svm_node;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;

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
    
    protected final List<svm_model> models;

    protected LibSVMModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo, boolean generatesProbabilities, List<svm_model> models) {
        super(name, description, featureIDMap, outputIDInfo, generatesProbabilities);
        this.models = models;
    }

    /**
     * Gets the underlying list of libsvm models.
     * @return The underlying model list.
     */
    public List<svm_model> getModel() {
        return models;
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

}
