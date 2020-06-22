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

package org.tribuo.regression.slm;

import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * A trainer for a lasso linear regression model which uses LARS to construct the model.
 * Each output dimension is trained independently.
 * <p>
 * See:
 * <pre>
 * Efron B, Hastie T, Johnstone I, Tibshirani R.
 * "Least Angle Regression"
 * The Annals of Statistics, 2004.
 * </pre>
 */
public class LARSLassoTrainer extends SLMTrainer {
    private static final Logger logger = Logger.getLogger(LARSLassoTrainer.class.getName());

    /**
     * Constructs a lasso LARS trainer for a linear model.
     * @param maxNumFeatures The maximum number of features to select. Supply -1 to select all features.
     */
    public LARSLassoTrainer(int maxNumFeatures) {
        super(true,maxNumFeatures);
    }

    /**
     * Constructs a lasso LARS trainer that selects all the features.
     */
    public LARSLassoTrainer() {
        this(-1);
    }

    @Override
    protected RealVector newWeights(SLMState state) {
        if (state.last) {
            return super.newWeights(state);
        }

        RealVector deltapi =  SLMTrainer.ordinaryLeastSquares(state.xpi,state.r);

        if (deltapi == null) {
            return null;
        }

        RealVector delta = state.unpack(deltapi);

        // Computing gamma
        List<Double> candidates = new ArrayList<>();

        double AA = SLMTrainer.sumInverted(state.xpi);
        double CC = state.C;

        RealVector wa = SLMTrainer.getwa(state.xpi,AA);
        RealVector ar = SLMTrainer.getA(state.X, state.xpi,wa);

        for (int i = 0; i < state.numFeatures; ++i) {
            if (!state.activeSet.contains(i)) {
                double c = state.corr.getEntry(i);
                double a = ar.getEntry(i);

                double v1 = (CC - c) / (AA - a);
                double v2 = (CC + c) / (AA + a);

                if (v1 >= 0) {
                    candidates.add(v1);
                }
                if (v2 >= 0) {
                    candidates.add(v2);
                }
            }
        }

        double gamma = Collections.min(candidates);

//        // The lasso modification
//        if (active.size() >= 2) {
//            int min = active.get(0);
//            double min_gamma = - beta.getEntry(min) / (wa.getEntry(active.indexOf(new Integer(min))) * (corr.getEntry(min) >= 0 ? +1 : -1));
//
//            for (int i = 1; i < active.size()-1; ++i) {
//                int idx = active.get(i);
//                double gamma_i = - beta.getEntry(idx) / (wa.getEntry(active.indexOf(new Integer(idx))) * (corr.getEntry(idx) >= 0 ? +1 : -1));
//                if (gamma_i < 0) continue;
//                if (gamma_i < min) {
//                    min = i;
//                    min_gamma = gamma_i;
//                }
//            }
//
//            if (min_gamma < gamma) {
//                active.remove(new Integer(min));
//                beta.setEntry(min,0.0);
//                return beta.add(delta.mapMultiplyToSelf(min_gamma));
//            }
//        }
//
//        return beta.add(delta.mapMultiplyToSelf(gamma));

        RealVector other = delta.mapMultiplyToSelf(gamma);

        for (int i = 0; i < state.numFeatures; ++i) {
            double betaElement = state.beta.getEntry(i);
            double otherElement = other.getEntry(i);
            if ((betaElement > 0 && betaElement + otherElement < 0)
                    || (betaElement < 0 && betaElement + otherElement > 0)) {
                state.beta.setEntry(i,0.0);
                other.setEntry(i,0.0);
                Integer integer = i;
                state.active.remove(integer);
                state.activeSet.remove(integer);
            }
        }

        return state.beta.add(other);
    }

    @Override
    public String toString() {
        return "LARSLassoTrainer(maxNumFeatures="+maxNumFeatures+")";
    }
}
