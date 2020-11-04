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

package org.tribuo.anomaly.evaluation;

import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.anomaly.Event;
import org.tribuo.anomaly.Event.EventType;
import org.tribuo.evaluation.metrics.EvaluationMetric;
import org.tribuo.evaluation.metrics.MetricContext;
import org.tribuo.evaluation.metrics.MetricTarget;

import java.util.List;
import java.util.function.ToDoubleBiFunction;

/**
 * A metric for evaluating anomaly detection problems. The sufficient statistics
 * must be encoded in the number of true positives, false positives, true negatives
 * and false negatives.
 */
public class AnomalyMetric implements EvaluationMetric<Event, AnomalyMetric.Context> {

    private final MetricTarget<Event> target;
    private final String name;

    private final ToDoubleBiFunction<MetricTarget<Event>, Context> impl;

    /**
     * Creates an anomaly detection metric, with a specific name, using the supplied evaluation function.
     * @param target The target of the metric (i.e., the event type or an average).
     * @param name The name of the metric.
     * @param impl The implementation function.
     */
    public AnomalyMetric(MetricTarget<Event> target, String name, ToDoubleBiFunction<MetricTarget<Event>, Context> impl) {
        this.target = target;
        this.name = name;
        this.impl = impl;
    }

    @Override
    public double compute(Context context) {
        return impl.applyAsDouble(target, context);
    }

    @Override
    public MetricTarget<Event> getTarget() {
        return target;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public Context createContext(Model<Event> model, List<Prediction<Event>> predictions) {
        return buildContext(model, predictions);
    }

    static Context buildContext(Model<Event> model, List<Prediction<Event>> predictions) {
        return new Context(model, predictions);
    }

    /**
     * The context for anomaly detection is the tp,fp,tn,fn statistics.
     */
    static final class Context extends MetricContext<Event> {

        // predicted anomalous, actually anomalous
        private final long truePositive;
        // predicted anomalous, actually expected
        private final long falsePositive;
        // predicted expected, actually expected
        private final long trueNegative;
        // predicted expected, actually anomalous
        private final long falseNegative;

        Context(Model<Event> model, List<Prediction<Event>> predictions) {
            super(model, predictions);
            PredictionStatistics tab = tabulate(predictions);
            truePositive = tab.truePositive;
            falsePositive = tab.falsePositive;
            trueNegative = tab.trueNegative;
            falseNegative = tab.falseNegative;
        }

        long getTruePositive() {
            return truePositive;
        }

        long getFalsePositive() {
            return falsePositive;
        }

        long getTrueNegative() {
            return trueNegative;
        }

        long getFalseNegative() {
            return falseNegative;
        }

        private static PredictionStatistics tabulate(List<Prediction<Event>> predictions) {
            // predicted anomalous, actually anomalous
            long truePositive = 0;
            // predicted anomalous, actually expected
            long falsePositive = 0;
            // predicted expected, actually expected
            long trueNegative = 0;
            // predicted expected, actually anomalous
            long falseNegative = 0;

            for (Prediction<Event> prediction : predictions) {
                Example<Event> example = prediction.getExample();
                Event.EventType truth = example.getOutput().getType();
                Event.EventType predicted = prediction.getOutput().getType();

                if (truth == EventType.ANOMALOUS) {
                    if (predicted == EventType.ANOMALOUS) {
                        truePositive++;
                    } else if (predicted == EventType.EXPECTED) {
                        falseNegative++;
                    } else {
                        //unknown predicted
                    }
                } else if (truth == EventType.EXPECTED) {
                    if (predicted == EventType.ANOMALOUS) {
                        falsePositive++;
                    } else if (predicted == EventType.EXPECTED) {
                        trueNegative++;
                    } else {
                        //unknown predicted
                    }
                } else {
                    // truth unknown
                    throw new IllegalArgumentException("Evaluation data contained EventType.UNKNOWN as the ground truth output.");
                }
            }
            return new PredictionStatistics(truePositive, falsePositive, trueNegative, falseNegative);
        }
    }

    /**
     * One day it will be a record. Not today though.
     */
    private static final class PredictionStatistics {
        private final long truePositive;
        private final long falsePositive;
        private final long trueNegative;
        private final long falseNegative;
        PredictionStatistics(long truePositive, long falsePositive, long trueNegative, long falseNegative) {
            this.truePositive = truePositive;
            this.falsePositive = falsePositive;
            this.trueNegative = trueNegative;
            this.falseNegative = falseNegative;
        }
    }

}