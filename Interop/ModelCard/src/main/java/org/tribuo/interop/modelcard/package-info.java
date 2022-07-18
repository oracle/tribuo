/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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
/**
 * ModelCard feature to allow more transparent model reporting.
 * <p>
 * See:
 * <pre>
 * M. Mitchell et al.
 * "Model Cards for Model Reporting"
 * In Conference in Fairness, Accountability, and Transparency, 2019.
 * </pre>
 * <p>
 * Using the ModelCard package, users can construct a {@link org.tribuo.interop.modelcard.ModelCard} object that partially documents
 * a trained model using Tribuo's built-in provenance. The remainder of the documentation
 * (i.e., adding extracted testing metrics and specifying the model's {@link org.tribuo.interop.modelcard.UsageDetails})
 * can either be left as blank, added by the user programmatically, or added by the user via
 * {@link org.tribuo.interop.modelcard.ModelCardCLI}.
 * <p>
 * A serialized ModelCard object with empty strings and lists set as the fields of {@link org.tribuo.interop.modelcard.UsageDetails} may
 * look like the following:
 * <pre>
 * {
 *   "ModelDetails" : {
 *     "schema-version" : "1.0",
 *     "model-type" : "LinearSGDModel",
 *     "model-package" : "org.tribuo.classification.sgd.linear.LinearSGDModel",
 *     "tribuo-version" : "4.3.0-SNAPSHOT",
 *     "java-version" : "17.0.2",
 *     "configured-parameters" : {
 *       "seed" : "12345",
 *       "tribuo-version" : "4.3.0-SNAPSHOT",
 *       "minibatchSize" : "1",
 *       "train-invocation-count" : "0",
 *       "is-sequence" : "false",
 *       "shuffle" : "true",
 *       "epochs" : "5",
 *       "optimiser" : {
 *         "epsilon" : "0.1",
 *         "initialLearningRate" : "1.0",
 *         "initialValue" : "0.0",
 *         "host-short-name" : "StochasticGradientOptimiser",
 *         "class-name" : "org.tribuo.math.optimisers.AdaGrad"
 *       },
 *       "host-short-name" : "Trainer",
 *       "class-name" : "org.tribuo.classification.sgd.linear.LogisticRegressionTrainer",
 *       "loggingInterval" : "1000",
 *       "objective" : {
 *         "host-short-name" : "LabelObjective",
 *         "class-name" : "org.tribuo.classification.sgd.objectives.LogMulticlass"
 *       }
 *     }
 *   },
 *   "TrainingDetails" : {
 *     "schema-version" : "1.0",
 *     "training-time" : "2022-07-16T14:21:11.683459-07:00",
 *     "training-set-size" : 7,
 *     "num-features" : 4,
 *     "features-list" : [ " a", " b", " c", " d" ],
 *     "num-outputs" : 2,
 *     "outputs-distribution" : {
 *       "a" : 2,
 *       "b" : 5
 *     }
 *   },
 *   "TestingDetails" : {
 *     "schema-version" : "1.0",
 *     "testing-set-size" : 3,
 *     "metrics" : {
 *       "overall-accuracy" : 0.3333333333333333,
 *       "average-precision" : 0.16666666666666666
 *     }
 *   },
 *   "UsageDetails" : {
 *     "schema-version" : "1.0",
 *     "intended-use" : "",
 *     "intended-users" : "",
 *     "out-of-scope-uses" : [ ],
 *     "pre-processing-steps" : [ ],
 *     "considerations-list" : [ ],
 *     "relevant-factors-list" : [ ],
 *     "resources-list" : [ ],
 *     "primary-contact" : "",
 *     "model-citation" : "",
 *     "model-license" : ""
 *   }
 * }
 * </pre>
 */
package org.tribuo.interop.modelcard;