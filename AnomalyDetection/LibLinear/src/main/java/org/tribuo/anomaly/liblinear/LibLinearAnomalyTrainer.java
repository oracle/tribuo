/*
 * Copyright (c) 2021, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.anomaly.liblinear;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.anomaly.Event;
import org.tribuo.anomaly.liblinear.LinearAnomalyType.LinearType;
import org.tribuo.common.liblinear.LibLinearModel;
import org.tribuo.common.liblinear.LibLinearTrainer;
import org.tribuo.provenance.ModelProvenance;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * A {@link Trainer} which wraps a liblinear-java anomaly detection trainer using a one-class SVM.
 * <p>
 * See:
 * <pre>
 * Fan RE, Chang KW, Hsieh CJ, Wang XR, Lin CJ.
 * "LIBLINEAR: A library for Large Linear Anomaly"
 * Journal of Machine Learning Research, 2008.
 * </pre>
 * and for the original algorithm:
 * <pre>
 * Cortes C, Vapnik V.
 * "Support-Vector Networks"
 * Machine Learning, 1995.
 * </pre>
 */
public class LibLinearAnomalyTrainer extends LibLinearTrainer<Event> {

    private static final Logger logger = Logger.getLogger(LibLinearAnomalyTrainer.class.getName());

    @Config(description = "Nu parameter in one class SVM.")
    private double nu = 0.5;

    /**
     * Creates a trainer using the default values (type:ONECLASS_SVM, cost:1, maxIterations:1000, terminationCriterion:0.1, nu:0.5).
     */
    public LibLinearAnomalyTrainer() {
        this(new LinearAnomalyType(LinearType.ONECLASS_SVM),1,1000,0.1, 0.5);
    }

    /**
     * Creates a trainer using the default values (type:ONECLASS_SVM, cost:1, maxIterations:1000, terminationCriterion:0.1) and the specified nu.
     * @param nu The nu parameter in the one-class SVM.
     */
    public LibLinearAnomalyTrainer(double nu) {
        this(new LinearAnomalyType(LinearType.ONECLASS_SVM),1,1000,0.1,nu);
    }

    /**
     * Creates a trainer for a LibLinearAnomalyModel. Sets maxIterations to 1000.
     * @param trainerType Loss function and optimisation method combination.
     * @param cost Cost penalty for each incorrectly classified training point.
     * @param terminationCriterion How close does the optimisation function need to be before terminating that subproblem (usually set to 0.1).
     * @param nu The nu parameter in the one-class SVM.
     */
    public LibLinearAnomalyTrainer(LinearAnomalyType trainerType, double cost, double terminationCriterion, double nu) {
        this(trainerType,cost,1000,terminationCriterion,nu);
    }

    /**
     * Creates a trainer for a LibLinear model
     * <p>
     * Uses {@link Trainer#DEFAULT_SEED} as the RNG seed.
     * @param trainerType Loss function and optimisation method combination.
     * @param cost Cost penalty for each incorrectly classified training point.
     * @param maxIterations The maximum number of dataset iterations.
     * @param terminationCriterion How close does the optimisation function need to be before terminating that subproblem (usually set to 0.1).
     * @param nu The nu parameter in the one-class SVM.
     */
    public LibLinearAnomalyTrainer(LinearAnomalyType trainerType, double cost, int maxIterations, double terminationCriterion, double nu) {
        this(trainerType,cost,maxIterations,terminationCriterion,nu,Trainer.DEFAULT_SEED);
    }

    /**
     * Creates a trainer for a LibLinear model
     * @param trainerType Loss function and optimisation method combination.
     * @param cost Cost penalty for each incorrectly classified training point.
     * @param maxIterations The maximum number of dataset iterations.
     * @param terminationCriterion How close does the optimisation function need to be before terminating that subproblem (usually set to 0.1).
     * @param nu The nu parameter in the one-class SVM.
     * @param seed The RNG seed.
     */
    public LibLinearAnomalyTrainer(LinearAnomalyType trainerType, double cost, int maxIterations, double terminationCriterion, double nu, long seed) {
        super(trainerType,cost,maxIterations,terminationCriterion,seed);
        this.nu = nu;
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        super.postConfig();
        if (!trainerType.isAnomaly()) {
            throw new IllegalArgumentException("Supplied classification or regression parameters to an anomaly detection linear model.");
        }
    }

    @Override
    protected Parameter setupParameters(ImmutableOutputInfo<Event> labelIDMap) {
        libLinearParams.setNu(nu);
        return libLinearParams.clone();
    }

    @Override
    protected List<Model> trainModels(Parameter curParams, int numFeatures, FeatureNode[][] features, double[][] outputs) {
        Problem data = new Problem();

        data.l = features.length;
        data.y = outputs[0];
        data.x = features;
        data.n = numFeatures;

        return Collections.singletonList(Linear.train(data,curParams));
    }

    @Override
    protected LibLinearModel<Event> createModel(ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Event> outputIDInfo, List<Model> models) {
        if (models.size() != 1) {
            throw new IllegalArgumentException("Anomaly detection uses a single model. Found " + models.size() + " models.");
        }
        return new LibLinearAnomalyModel("liblinear-anomaly-model",provenance,featureIDMap,outputIDInfo,models);
    }

    @Override
    protected Pair<FeatureNode[][], double[][]> extractData(Dataset<Event> data, ImmutableOutputInfo<Event> outputInfo, ImmutableFeatureMap featureMap) {
        ArrayList<FeatureNode> featureCache = new ArrayList<>();
        FeatureNode[][] features = new FeatureNode[data.size()][];
        double[][] outputs = new double[1][data.size()];
        int i = 0;
        for (Example<Event> e : data) {
            outputs[0][i] = outputInfo.getID(e.getOutput());
            features[i] = exampleToNodes(e,featureMap,featureCache);
            i++;
        }
        return new Pair<>(features,outputs);
    }

}
