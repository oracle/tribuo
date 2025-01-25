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

package org.tribuo.regression.sgd.linear;

import com.oracle.labs.mlrg.olcut.config.Config;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.common.sgd.AbstractLinearTrainer;
import org.tribuo.common.sgd.SGDObjective;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.sgd.RegressionObjective;
import org.tribuo.regression.sgd.objectives.SquaredLoss;

import java.util.logging.Logger;

/**
 * A trainer for a linear regression using L-BFGS.
 * <p>
 * See:
 * <pre>
 * Nocedal, J. and Wright, S.
 * "Numerical Optimization (2nd Edition)"
 * Springer, 2006.
 * </pre>
 */
public final class LinearTrainer  extends AbstractLinearTrainer<Regressor, DenseMatrix, LinearSGDModel> {
    private static final Logger logger = Logger.getLogger(LinearTrainer.class.getName());

    @Config(description = "The regression objective function to use.")
    private RegressionObjective objective = new SquaredLoss();


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
    public LinearTrainer(RegressionObjective objective, int maxIterations, boolean l2Penalty, double tolerance, double gradientTolerance, double regularisationStrength) {
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
    public void postConfig() {}

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
    protected DenseMatrix createTargets(Dataset<Regressor> dataset, ImmutableOutputInfo<Regressor> outputInfo) {
        DenseVector[] sgdTargets = new DenseVector[dataset.size()];
        int n = 0;
        for (Example<Regressor> example : dataset) {
            double[] regressorsBuffer = new double[outputInfo.size()];
            for (Regressor.DimensionTuple r : example.getOutput()) {
                int id = outputInfo.getID(r);
                regressorsBuffer[id] = r.getValue();
            }
            sgdTargets[n] = DenseVector.createDenseVector(regressorsBuffer);
            n++;
        }
        return DenseMatrix.createDenseMatrix(sgdTargets);
    }

    @Override
    protected SGDObjective<DenseVector, DenseMatrix> getObjective() {
        return objective;
    }

    @Override
    protected LinearSGDModel createModel(ModelProvenance provenance, ImmutableFeatureMap featureMap, ImmutableOutputInfo<Regressor> outputInfo, LinearParameters parameters) {
        String[] dimensionNames = new String[outputInfo.size()];
        for (Regressor r : outputInfo.getDomain()) {
            int id = outputInfo.getID(r);
            dimensionNames[id] = r.getNames()[0];
        }
        return new LinearSGDModel("linear-lbfgs-model", dimensionNames, provenance, featureMap, outputInfo, parameters);
    }

    @Override
    protected String getModelClassName() {
        return LinearSGDModel.class.getName();
    }
}
