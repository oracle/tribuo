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

package org.tribuo.classification.liblinear;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.WeightedLabels;
import org.tribuo.classification.liblinear.LinearClassificationType.LinearType;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * A {@link Trainer} which wraps a liblinear-java classifier trainer.
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
public class LibLinearClassificationTrainer extends LibLinearTrainer<Label> implements WeightedLabels {

    private static final Logger logger = Logger.getLogger(LibLinearClassificationTrainer.class.getName());

    @Config(description="Use Label specific weights.")
    private Map<String,Float> labelWeights = Collections.emptyMap();

    /**
     * Creates a trainer using the default values ({@link LinearType#L2R_L2LOSS_SVC_DUAL}, 1, 0.1, {@link Trainer#DEFAULT_SEED}).
     */
    public LibLinearClassificationTrainer() {
        this(new LinearClassificationType(LinearType.L2R_L2LOSS_SVC_DUAL),1,1000,0.1);
    }

    /**
     * Creates a trainer for a LibLinearClassificationModel.
     * <p>
     * Uses {@link Trainer#DEFAULT_SEED} as the RNG seed. Sets maxIterations to 1000.
     * @param trainerType Loss function and optimisation method combination.
     * @param cost Cost penalty for each incorrectly classified training point.
     * @param terminationCriterion How close does the optimisation function need to be before terminating that subproblem (usually set to 0.1).
     */
    public LibLinearClassificationTrainer(LinearClassificationType trainerType, double cost, double terminationCriterion) {
        this(trainerType,cost,1000,terminationCriterion);
    }

    /**
     * Creates a trainer for a LibLinear model
     * <p>
     * Uses {@link Trainer#DEFAULT_SEED} as the RNG seed.
     * @param trainerType Loss function and optimisation method combination.
     * @param cost Cost penalty for each incorrectly classified training point.
     * @param maxIterations The maximum number of dataset iterations.
     * @param terminationCriterion How close does the optimisation function need to be before terminating that subproblem (usually set to 0.1).
     */
    public LibLinearClassificationTrainer(LinearClassificationType trainerType, double cost, int maxIterations, double terminationCriterion) {
        this(trainerType,cost,maxIterations,terminationCriterion,Trainer.DEFAULT_SEED);
    }

    /**
     * Creates a trainer for a LibLinear model
     * @param trainerType Loss function and optimisation method combination.
     * @param cost Cost penalty for each incorrectly classified training point.
     * @param maxIterations The maximum number of dataset iterations.
     * @param terminationCriterion How close does the optimisation function need to be before terminating that subproblem (usually set to 0.1).
     * @param seed The RNG seed.
     */
    public LibLinearClassificationTrainer(LinearClassificationType trainerType, double cost, int maxIterations, double terminationCriterion, long seed) {
        super(trainerType,cost,maxIterations,terminationCriterion,seed);
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        super.postConfig();
        if (!trainerType.isClassification()) {
            throw new IllegalArgumentException("Supplied regression or anomaly detection parameters to a classification linear model.");
        }
    }

    @Override
    protected List<Model> trainModels(Parameter curParams, int numFeatures, FeatureNode[][] features, double[][] outputs) {
        Problem data = new Problem();

        data.l = features.length;
        data.y = outputs[0];
        data.x = features;
        data.n = numFeatures;
        data.bias = 1.0;

        return Collections.singletonList(Linear.train(data,curParams));
    }

    @Override
    protected LibLinearModel<Label> createModel(ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> outputIDInfo, List<Model> models) {
        if (models.size() != 1) {
            throw new IllegalArgumentException("Classification uses a single model. Found " + models.size() + " models.");
        }
        return new LibLinearClassificationModel("liblinear-classification-model",provenance,featureIDMap,outputIDInfo,models);
    }

    @Override
    protected Pair<FeatureNode[][], double[][]> extractData(Dataset<Label> data, ImmutableOutputInfo<Label> outputInfo, ImmutableFeatureMap featureMap) {
        ArrayList<FeatureNode> featureCache = new ArrayList<>();
        FeatureNode[][] features = new FeatureNode[data.size()][];
        double[][] outputs = new double[1][data.size()];
        int i = 0;
        for (Example<Label> e : data) {
            outputs[0][i] = outputInfo.getID(e.getOutput());
            features[i] = exampleToNodes(e,featureMap,featureCache);
            i++;
        }
        return new Pair<>(features,outputs);
    }

    @Override
    protected Parameter setupParameters(ImmutableOutputInfo<Label> labelIDMap) {
        Parameter curParams = libLinearParams.clone();
        if (!labelWeights.isEmpty()) {
            double[] weights = new double[labelIDMap.size()];
            int[] indices = new int[labelIDMap.size()];
            int i = 0;
            for (Pair<Integer,Label> label : labelIDMap) {
                String labelName = label.getB().getLabel();
                Float weight = labelWeights.get(labelName);
                indices[i] = label.getA();
                if (weight != null) {
                    weights[i] = weight;
                } else {
                    weights[i] = 1.0f;
                }
                i++;
            }
            curParams.setWeights(weights,indices);
            //logger.info("Weights = " + Arrays.toString(weights) + ", labels = " + Arrays.toString(indices) + ", outputIDInfo = " + outputIDInfo);
        }
        return curParams;
    }

    @Override
    public void setLabelWeights(Map<Label,Float> weights) {
        labelWeights = new HashMap<>();
        for (Map.Entry<Label,Float> e : weights.entrySet()) {
            labelWeights.put(e.getKey().getLabel(),e.getValue());
        }
    }

}
