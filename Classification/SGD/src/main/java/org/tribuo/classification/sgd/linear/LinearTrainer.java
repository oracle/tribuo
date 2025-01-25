/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.sgd.linear;

import com.oracle.labs.mlrg.olcut.config.Config;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.sgd.LabelObjective;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.common.sgd.AbstractLinearTrainer;
import org.tribuo.common.sgd.SGDObjective;
import org.tribuo.math.LinearParameters;
import org.tribuo.provenance.ModelProvenance;

import java.util.logging.Logger;

/**
 * A trainer for a linear classifier using L-BFGS.
 * <p>
 * See:
 * <pre>
 * Nocedal, J. and Wright, S.
 * "Numerical Optimization (2nd Edition)"
 * Springer, 2006.
 * </pre>
 */
public final class LinearTrainer extends AbstractLinearTrainer<Label, int[], LinearSGDModel> {
    private static final Logger logger = Logger.getLogger(LinearTrainer.class.getName());

    @Config(description = "The classification objective function to use.")
    private LabelObjective objective = new LogMulticlass();

    /**
     * Constructs a trainer for a linear model using L-BFGS.
     *
     * @param objective       The objective function to optimise.
     * @param maxIterations The maximum number of L-BFGS iterations.
     * @param l2Penalty Should it use L2 regularisation to fit the model.
     * @param tolerance Convergence tolerance on the loss.
     * @param gradientTolerance Convergence tolerance on the gradient.
     * @param regularisationStrength Strength of the L2 regularisation penalty term.
     */
    public LinearTrainer(LabelObjective objective, int maxIterations, boolean l2Penalty, double tolerance, double gradientTolerance, double regularisationStrength) {
        super(maxIterations, l2Penalty, tolerance, gradientTolerance, regularisationStrength);
        this.objective = objective;
        postConfig();
    }

    /**
     * For OLCUT.
     */
    private LinearTrainer() {
        super();
    }

    @Override
    public void postConfig() { }

    @Override
    public String toString() {
        return "LinearTrainer(" +
                "objective=" + objective +
                ", maxIterations=" + maxIterations +
                ", l2Penalty=" + l2Penalty +
                ", tolerance=" + tolerance +
                ", gradientTolerance=" + gradientTolerance +
                ", regularisationStrength=" + regularisationStrength +
                ", memorySize=" + memorySize +
                ')';
    }

    @Override
    protected int[] createTargets(Dataset<Label> dataset, ImmutableOutputInfo<Label> outputInfo) {
        int[] outputs = new int[dataset.size()];

        int i = 0;
        for (Example<Label> e : dataset) {
            outputs[i] = outputInfo.getID(e.getOutput());
            i++;
        }
        return outputs;
    }

    @Override
    protected SGDObjective<Integer, int[]> getObjective() {
        return objective;
    }

    @Override
    protected LinearSGDModel createModel(ModelProvenance provenance, ImmutableFeatureMap featureMap, ImmutableOutputInfo<Label> outputInfo, LinearParameters parameters) {
        return new LinearSGDModel("linear-lbfgs-model", provenance, featureMap, outputInfo, parameters, objective.getNormalizer(), objective.isProbabilistic());
    }

    @Override
    protected String getModelClassName() {
        return LinearSGDModel.class.getName();
    }
}
