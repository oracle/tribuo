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

package org.tribuo.anomaly.libsvm;

import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.anomaly.Event;
import org.tribuo.anomaly.Event.EventType;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.LibSVMTrainer;
import org.tribuo.common.libsvm.SVMParameters;
import org.tribuo.provenance.ModelProvenance;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * A trainer for anomaly models that uses LibSVM.
 * <p>
 * See:
 * <pre>
 * Chang CC, Lin CJ.
 * "LIBSVM: a library for Support Vector Machines"
 * ACM transactions on intelligent systems and technology (TIST), 2011.
 * </pre>
 * <p>
 * and for the anomaly detection algorithm:
 * <pre>
 * Schölkopf B, Platt J, Shawe-Taylor J, Smola A J, Williamson R C.
 * "Estimating the support of a high-dimensional distribution"
 * Neural Computation, 2001, 1443-1471.
 * </pre>
 */
public class LibSVMAnomalyTrainer extends LibSVMTrainer<Event> {
    private static final Logger logger = Logger.getLogger(LibSVMAnomalyTrainer.class.getName());

    /**
     * For OLCUT.
     */
    protected LibSVMAnomalyTrainer() {}

    /**
     * Creates a one-class LibSVM trainer using the supplied parameters.
     * @param parameters The training parameters.
     */
    public LibSVMAnomalyTrainer(SVMParameters<Event> parameters) {
        super(parameters);
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        super.postConfig();
        if (!svmType.isAnomaly()) {
            throw new IllegalArgumentException("Supplied classification or regression parameters to an anomaly detection SVM.");
        }
    }

    @Override
    public LibSVMModel<Event> train(Dataset<Event> dataset, Map<String, Provenance> instanceProvenance) {
        for (Pair<String,Long> p : dataset.getOutputInfo().outputCountsIterable()) {
            if (p.getA().equals(EventType.ANOMALOUS.toString()) && (p.getB() > 0)) {
                throw new IllegalArgumentException("LibSVMAnomalyTrainer only supports EXPECTED events at training time.");
            }
        }
        return super.train(dataset,instanceProvenance);
    }

    @Override
    protected LibSVMModel<Event> createModel(ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Event> outputIDInfo, List<svm_model> models) {
        return new LibSVMAnomalyModel("svm-anomaly-detection-model", provenance, featureIDMap, outputIDInfo, models);
    }

    @Override
    protected List<svm_model> trainModels(svm_parameter curParams, int numFeatures, svm_node[][] features, double[][] outputs) {
        svm_problem problem = new svm_problem();
        problem.l = outputs[0].length;
        problem.x = features;
        problem.y = outputs[0];
        if (curParams.gamma == 0) {
            curParams.gamma = 1.0 / numFeatures;
        }
        String checkString = svm.svm_check_parameter(problem, curParams);
        if(checkString != null) {
            throw new IllegalArgumentException("Error checking SVM parameters: " + checkString);
        }
        return Collections.singletonList(svm.svm_train(problem, curParams));
    }

    @Override
    protected Pair<svm_node[][], double[][]> extractData(Dataset<Event> data, ImmutableOutputInfo<Event> outputInfo, ImmutableFeatureMap featureMap) {
        double[][] ys = new double[1][data.size()];
        svm_node[][] xs = new svm_node[data.size()][];
        List<svm_node> buffer = new ArrayList<>();
        int i = 0;
        for (Example<Event> example : data) {
            ys[0][i] = extractOutput(example.getOutput());
            xs[i] = exampleToNodes(example, featureMap, buffer);
            i++;
        }
        return new Pair<>(xs,ys);
    }

    /**
     * Converts an output into a double for use in training.
     * <p>
     * By convention {@link EventType#EXPECTED} is 1.0, other events are -1.0.
     * @param output The output to convert.
     * @return The double value.
     */
    protected double extractOutput(Event output) {
        if (output.getType() == Event.EventType.EXPECTED) {
            return 1.0;
        } else {
            return -1.0;
        }
    }
}
