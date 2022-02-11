/*
 * Copyright (c) 2020, 2022 Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.xgboost;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static ml.dmlc.xgboost4j.java.Booster.FeatureImportanceType.*;

/**
 * Generate and collate feature importance information from the XGBoost model. This wraps the underlying functionality
 * of the XGBoost model, and should provide feature importance metrics compatible with those provided by XGBoost's R
 * and Python APIs. For a more treatment of what the different importance metrics mean and how to interpret them, see
 * <a href="https://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html">here</a>. In brief
 *
 * <ul>
 *     <li><b>Gain</b> measures the improvement in accuracy that a feature brings to the branches on which it appears.
 *     This represents the sum of situated marginal contributions that a given feature makes to the each branching
 *     chain in which it appears.</li>
 *     <li><b>Cover</b> measures the number of examples a given feature discriminates across, relative to the total
 *     number of examples all features discriminate across.</li>
 *     <li><b>Weight</b> measures the number a times a feature occurs in the model. Due to the way the model builds trees,
 *     this value is skewed in favor of continuous features.</li>
 *     <li><b>Total Gain</b> is similar to gain, but not locally averaged by weight, and thus not skewed in the way that
 *     weight can be skewed.</li>
 *     <li><b>Total Cover</b> is similar to cover, but not locally averaged by weight, and thus not skewed in the way that
 *     weight can be skewed.</li>
 * </ul>
 */
public class XGBoostFeatureImportance {

    /**
     * An instance of feature importance values for a single feature. See {@link XGBoostFeatureImportance} for details
     * on interpreting the metrics.
     */
    public static class XGBoostFeatureImportanceInstance {

        private final String featureName;
        private final double gain;
        private final double cover;
        private final double weight;
        private final double totalGain;
        private final double totalCover;

        XGBoostFeatureImportanceInstance(String featureName, double gain, double cover, double weight, double totalGain, double totalCover) {
            this.featureName = featureName;
            this.gain = gain;
            this.cover = cover;
            this.weight = weight;
            this.totalGain = totalGain;
            this.totalCover = totalCover;
        }

        /**
         * The feature name.
         * @return The feature name.
         */
        public String getFeatureName() {
            return featureName;
        }

        /**
         * The information gain a feature provides when split on.
         * @return The gain.
         */
        public double getGain() {
            return gain;
        }

        /**
         * The number of examples a feature discriminates between.
         * @return The cover.
         */
        public double getCover() {
            return cover;
        }

        /**
         * The number of times a feature is used in the model.
         * @return The number of times the feature is used to split.
         */
        public double getWeight() {
            return weight;
        }

        /**
         * The total gain across all times the feature is used to split.
         * @return The total gain.
         */
        public double getTotalGain() {
            return totalGain;
        }

        /**
         * The total number of examples a feature discrimnates between.
         * @return The total cover.
         */
        public double getTotalCover() {
            return totalCover;
        }

        @Override
        public String toString() {
            return String.format("XGBoostFeatureImportanceRecord(feature=%s, gain=%.2f, cover=%.2f, weight=%.2f, totalGain=%.2f, totalCover=%.2f)",
                    featureName, gain, cover, weight, totalGain, totalCover);
        }
    }

    private final Booster booster;
    private final ImmutableFeatureMap featureMap;
    private final Model<?> model;

    XGBoostFeatureImportance(Booster booster, Model<?> model) {
        this.booster = booster;
        this.model = model;
        this.featureMap = model.getFeatureIDMap();
    }

    private String translateFeatureId(String xgbFeatName) {
        return featureMap.get(Integer.parseInt(xgbFeatName.substring(1))).getName();
    }

    private Stream<Map.Entry<String, Double>> getImportanceStream(String importanceType) {
        try {
            return booster.getScore("", importanceType).entrySet().stream()
                    .sorted(Comparator.comparingDouble((Map.Entry<String, Double> e) -> e.getValue()).reversed());
        } catch (XGBoostError e) {
            throw new IllegalStateException("Error generating feature importance for " + importanceType + " caused by", e);
        }
    }

    private LinkedHashMap<String, Double> coalesceImportanceStream(Stream<Map.Entry<String, Double>> str) {
        return str.collect(Collectors.toMap(e -> translateFeatureId(e.getKey()),
                Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
    }

    /**
     * Gain measures the improvement in accuracy that a feature brings to the branches on which it appears.
     * This represents the sum of situated marginal contributions that a given feature makes to the each branching
     * chain in which it appears.
     * @return Ordered map where the keys are feature names and the value is the gain, sorted descending
     */
    public LinkedHashMap<String, Double> getGain() {
        return coalesceImportanceStream(getImportanceStream(GAIN));
    }

    /**
     *
     * Gain measures the improvement in accuracy that a feature brings to the branches on which it appears.
     * This represents the sum of situated marginal contributions that a given feature makes to the each branching
     * chain in which it appears. Returns only the top numFeatures features.
     * @param numFeatures number of features to return
     * @return Ordered map where the keys are feature names and the value is the gain, sorted descending
     */
    public LinkedHashMap<String, Double> getGain(int numFeatures) {
        return coalesceImportanceStream(getImportanceStream(GAIN).limit(numFeatures));
    }

    /**
     * Cover measures the number of examples a given feature discriminates across, relative to the total
     * number of examples all features discriminate across.
     * @return Ordered map where the keys are feature names and the value is the cover, sorted descending
     */
    public LinkedHashMap<String, Double> getCover() {
        return coalesceImportanceStream(getImportanceStream(COVER));
    }
    /**
     *
     * Cover measures the number of examples a given feature discriminates across, relative to the total.
     * number of examples all features discriminate across. Returns only the top numFeatures features.
     * @param numFeatures number of features to return
     * @return Ordered map where the keys are feature names and the value is the cover, sorted descending
     */
    public LinkedHashMap<String, Double> getCover(int numFeatures) {
        return coalesceImportanceStream(getImportanceStream(COVER).limit(numFeatures));
    }

    /**
     * Weight measures the number a times a feature occurs in the model. Due to the way the model builds trees,
     * this value is skewed in favor of continuous features.
     * @return Ordered map where the keys are feature names and the value is the weight, sorted descending
     */
    public LinkedHashMap<String, Double> getWeight() {
        return coalesceImportanceStream(getImportanceStream(WEIGHT));
    }
    /**
     * Weight measures the number a times a feature occurs in the model. Due to the way the model builds trees,
     * this value is skewed in favor of continuous features. Returns only the top numFeatures features.
     * @param numFeatures number of features to return
     * @return Ordered map where the keys are feature names and the value is the weight, sorted descending
     */
    public LinkedHashMap<String, Double> getWeight(int numFeatures) {
        return coalesceImportanceStream(getImportanceStream(WEIGHT).limit(numFeatures));
    }

    /**
     * Total Gain is similar to gain, but not locally averaged by weight, and thus not skewed in the way that
     * weight can be skewed.
     * @return Ordered map where the keys are feature names and the value is the total gain, sorted descending
     */
    public LinkedHashMap<String, Double> getTotalGain() {
        return coalesceImportanceStream(getImportanceStream(TOTAL_GAIN));
    }
    /**
     * Total Gain is similar to gain, but not locally averaged by weight, and thus not skewed in the way that
     * weight can be skewed. Returns only top numFeatures features.
     * @param numFeatures number of features to return
     * @return Ordered map where the keys are feature names and the value is the total gain, sorted descending
     */
    public LinkedHashMap<String, Double> getTotalGain(int numFeatures) {
        return coalesceImportanceStream(getImportanceStream(TOTAL_GAIN).limit(numFeatures));
    }

    /**
     * Total Cover is similar to cover, but not locally averaged by weight, and thus not skewed in the way that
     * weight can be skewed.
     * @return Ordered map where the keys are feature names and the value is the total gain, sorted descending
     */
    public LinkedHashMap<String, Double> getTotalCover() {
        return coalesceImportanceStream(getImportanceStream(TOTAL_COVER));
    }
    /**
     * Total Cover is similar to cover, but not locally averaged by weight, and thus not skewed in the way that
     * weight can be skewed. Returns only top numFeatures features.
     * @param numFeatures number of features to return
     * @return Ordered map where the keys are feature names and the value is the total gain, sorted descending
     */
    public LinkedHashMap<String, Double> getTotalCover(int numFeatures) {
        return coalesceImportanceStream(getImportanceStream(TOTAL_COVER).limit(numFeatures));
    }

    /**
     * Gets all the feature importances for all the features.
     * @return records of all importance metrics for each feature, sorted by gain.
     */
    public List<XGBoostFeatureImportanceInstance> getImportances() {
        Map<String, LinkedHashMap<String, Double>> importanceByType = Stream.of(GAIN, COVER, WEIGHT, TOTAL_GAIN, TOTAL_COVER)
                .map(importanceType -> new AbstractMap.SimpleEntry<>(importanceType, coalesceImportanceStream(getImportanceStream(importanceType))))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        // this is already sorted by gain
        List<String> features = new ArrayList<>(importanceByType.get(GAIN).keySet());
        return features.stream().map(featureName -> new XGBoostFeatureImportanceInstance(featureName,
                importanceByType.get(GAIN).get(featureName),
                importanceByType.get(COVER).get(featureName),
                importanceByType.get(WEIGHT).get(featureName),
                importanceByType.get(TOTAL_GAIN).get(featureName),
                importanceByType.get(TOTAL_COVER).get(featureName)))
                .collect(Collectors.toList());
    }

    /**
     * Gets the feature importances for the top n features sorted by gain.
     * @param numFeatures number of features to return
     * @return records of all importance metrics for each feature, sorted by gain.
     */
    public List<XGBoostFeatureImportanceInstance> getImportances(int numFeatures) {
        Map<String, LinkedHashMap<String, Double>> importanceByType = Stream.of(GAIN, COVER, WEIGHT, TOTAL_GAIN, TOTAL_COVER)
                .map(importanceType -> new AbstractMap.SimpleEntry<>(importanceType, coalesceImportanceStream(getImportanceStream(importanceType))))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        // this is already sorted by gain
        List<String> features = new ArrayList<>(importanceByType.get(GAIN).keySet()).subList(0, Math.min(importanceByType.get(GAIN).keySet().size(), numFeatures));
        return features.stream().map(featureName -> new XGBoostFeatureImportanceInstance(featureName,
                importanceByType.get(GAIN).get(featureName),
                importanceByType.get(COVER).get(featureName),
                importanceByType.get(WEIGHT).get(featureName),
                importanceByType.get(TOTAL_GAIN).get(featureName),
                importanceByType.get(TOTAL_COVER).get(featureName)))
                .collect(Collectors.toList());
    }

    @Override
    public String toString() {
        return String.format("XGBoostFeatureImportance(model=%s)", model.toString());
    }
}
