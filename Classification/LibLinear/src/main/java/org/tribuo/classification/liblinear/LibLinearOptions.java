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

package org.tribuo.classification.liblinear;

import com.oracle.labs.mlrg.olcut.config.Option;
import org.tribuo.classification.ClassificationOptions;
import org.tribuo.classification.liblinear.LinearClassificationType.LinearType;

/**
 * Command line options for working with a classification liblinear model.
 */
public class LibLinearOptions implements ClassificationOptions<LibLinearClassificationTrainer> {

    @Override
    public String getOptionsDescription() {
        return "Options for parameterising a LibLinear classification trainer.";
    }

    @Option(longName = "liblinear-solver-type", usage = "Type of linear model, defaults to L2R_L2LOSS_SVC_DUAL.")
    LinearType liblinearSolverType = LinearType.L1R_LR;

    @Option(longName = "liblinear-cost", usage = "cost")
    double liblinearCost = 1.0d;

    @Option(longName = "liblinear-eps", usage = "stopping criteria")
    double liblinearEps = 0.01d; //TODO or 0.1?

    @Option(longName = "liblinear-maxiters", usage = "max iterations")
    int liblinearMaxiters = 1000;

    @Override
    public LibLinearClassificationTrainer getTrainer() {
        return new LibLinearClassificationTrainer(new LinearClassificationType(liblinearSolverType), liblinearCost, liblinearMaxiters, liblinearEps);
    }
}
