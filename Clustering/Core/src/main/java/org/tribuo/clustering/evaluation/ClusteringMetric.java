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

package org.tribuo.clustering.evaluation;

import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.clustering.ClusterID;
import org.tribuo.evaluation.metrics.EvaluationMetric;
import org.tribuo.evaluation.metrics.MetricContext;
import org.tribuo.evaluation.metrics.MetricTarget;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

/**
 * A metric for evaluating clustering problems. The sufficient statistics are the cluster
 * ids assigned to every point, along with the "true" ids.
 */
public class ClusteringMetric implements EvaluationMetric<ClusterID, ClusteringMetric.Context> {

    private final MetricTarget<ClusterID> target;
    private final String name;
    private final BiFunction<MetricTarget<ClusterID>, Context, Double> impl;

    public ClusteringMetric(MetricTarget<ClusterID> target, String name, BiFunction<MetricTarget<ClusterID>, Context, Double> impl) {
        this.target = target;
        this.name = name;
        this.impl = impl;
    }

    @Override
    public double compute(Context context) {
        return impl.apply(target, context);
    }

    @Override
    public MetricTarget<ClusterID> getTarget() {
        return target;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public Context createContext(Model<ClusterID> model, List<Prediction<ClusterID>> predictions) {
        return buildContext(model, predictions);
    }

    @Override
    public String toString() {
        return "ClusteringMetric(" +
                "target=" + target +
                ",name='" + name + '\'' +
                ')';
    }

    static final class Context extends MetricContext<ClusterID> {

        private final ArrayList<Integer> predictedIDs = new ArrayList<>();
        private final ArrayList<Integer> trueIDs = new ArrayList<>();

        Context(Model<ClusterID> model, List<Prediction<ClusterID>> predictions) {
            super(model, predictions);
            for (Prediction<ClusterID> pred : predictions) {
                predictedIDs.add(pred.getOutput().getID());
                trueIDs.add(pred.getExample().getOutput().getID());
            }
        }

        public ArrayList<Integer> getPredictedIDs() {
            return predictedIDs;
        }

        public ArrayList<Integer> getTrueIDs() {
            return trueIDs;
        }
    }

    static Context buildContext(Model<ClusterID> model, List<Prediction<ClusterID>> predictions) {
        return new Context(model, predictions);
    }


}