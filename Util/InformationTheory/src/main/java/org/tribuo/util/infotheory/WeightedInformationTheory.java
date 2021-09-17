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

package org.tribuo.util.infotheory;

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.util.infotheory.impl.CachedPair;
import org.tribuo.util.infotheory.impl.CachedTriple;
import org.tribuo.util.infotheory.impl.PairDistribution;
import org.tribuo.util.infotheory.impl.TripleDistribution;
import org.tribuo.util.infotheory.impl.WeightCountTuple;
import org.tribuo.util.infotheory.impl.WeightedPairDistribution;
import org.tribuo.util.infotheory.impl.WeightedTripleDistribution;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A class of (discrete) weighted information theoretic functions. Gives warnings if
 * there are insufficient samples to estimate the quantities accurately.
 * <p>
 * Defaults to log_2, so returns values in bits.
 * <p>
 * All functions expect that the element types have well defined equals and
 * hashcode, and that equals is consistent with hashcode. The behaviour is undefined
 * if this is not true.
 */
public final class WeightedInformationTheory {
    private static final Logger logger = Logger.getLogger(WeightedInformationTheory.class.getName());

    /**
     * The ratio of samples to symbols before emitting a warning.
     */
    public static final double SAMPLES_RATIO = 5.0;
    /**
     * The initial size of the various maps.
     */
    public static final int DEFAULT_MAP_SIZE = 20;
    /**
     * Log base 2.
     */
    public static final double LOG_2 = Math.log(2);
    /**
     * Log base e.
     */
    public static final double LOG_E = Math.log(Math.E);

    /**
     * Sets the base of the logarithm used in the information theoretic calculations.
     * For LOG_2 the unit is "bit", for LOG_E the unit is "nat".
     */
    public static double LOG_BASE = LOG_2;

    /**
     * Chooses which variable is the one with associated weights.
     */
    public enum VariableSelector {
        /**
         * The first variable is weighted.
         */
        FIRST,
        /**
         * The second variable is weighted.
         */
        SECOND,
        /**
         * The third variable is weighted.
         */
        THIRD
    }

    /**
     * Private constructor, only has static methods.
     */
    private WeightedInformationTheory() {}

    /**
     * Calculates the discrete weighted joint mutual information, using
     * histogram probability estimators. Arrays must be the same length.
     * @param <T1> Type contained in the first array.
     * @param <T2> Type contained in the second array.
     * @param <T3> Type contained in the target array.
     * @param first An array of values.
     * @param second Another array of values.
     * @param target Target array of values.
     * @param weights Array of weight values.
     * @return The weighted mutual information I_w(first,second;joint)
     */
    public static <T1,T2,T3> double jointMI(List<T1> first, List<T2> second, List<T3> target, List<Double> weights) {
        WeightedTripleDistribution<T1, T2, T3> tripleRV = WeightedTripleDistribution.constructFromLists(first, second, target, weights);

        return jointMI(tripleRV);
    }

    /**
     * Calculates the discrete weighted joint mutual information, using
     * histogram probability estimators.
     * @param tripleRV The weighted triple distribution.
     * @param <T1> The first element type.
     * @param <T2> The second element type.
     * @param <T3> The third element type.
     * @return The weighted mutual information I_w(first,second;joint)
     */
    public static <T1,T2,T3> double jointMI(WeightedTripleDistribution<T1,T2,T3> tripleRV) {
        Map<CachedTriple<T1,T2,T3>, WeightCountTuple> jointCount = tripleRV.getJointCount();
        Map<CachedPair<T1,T2>,WeightCountTuple> abCount = tripleRV.getABCount();
        Map<T3,WeightCountTuple> cCount = tripleRV.getCCount();

        double vectorLength = tripleRV.count;
        double jmi = 0.0;
        for (Entry<CachedTriple<T1,T2,T3>,WeightCountTuple> e : jointCount.entrySet()) {
            double jointCurCount = e.getValue().count;
            double jointCurWeight = e.getValue().weight;
            double prob = jointCurCount / vectorLength;
            CachedPair<T1,T2> pair = e.getKey().getAB();
            double abCurCount = abCount.get(pair).count;
            double cCurCount = cCount.get(e.getKey().getC()).count;

            jmi += jointCurWeight * prob * Math.log((vectorLength*jointCurCount)/(abCurCount*cCurCount));
        }
        jmi /= LOG_BASE;

        double stateRatio = vectorLength / jointCount.size();
        if (stateRatio < SAMPLES_RATIO) {
            logger.log(Level.INFO, "Joint MI estimate of {0} had samples/state ratio of {1}", new Object[]{jmi, stateRatio});
        }
        
        return jmi;
    }

    /**
     * Calculates the discrete weighted joint mutual information, using
     * histogram probability estimators.
     * @param rv The triple distribution.
     * @param weights The weights for one of the variables.
     * @param vs The weighted variable id.
     * @param <T1> The first element type.
     * @param <T2> The second element type.
     * @param <T3> The third element type.
     * @return The weighted mutual information I_w(first,second;joint)
     */
    public static <T1,T2,T3> double jointMI(TripleDistribution<T1,T2,T3> rv, Map<?,Double> weights, VariableSelector vs){
        Double boxedWeight;
        double vecLength = rv.count;
        Map<CachedTriple<T1,T2,T3>,MutableLong> jointCount = rv.getJointCount();
        Map<CachedPair<T1,T2>,MutableLong> abCount = rv.getABCount();
        Map<T3,MutableLong> cCount = rv.getCCount();

        double jmi = 0.0;
        for (Entry<CachedTriple<T1,T2,T3>,MutableLong> e : jointCount.entrySet()) {
            double jointCurCount = e.getValue().doubleValue();
            double prob = jointCurCount / vecLength;
            CachedPair<T1,T2> pair = new CachedPair<>(e.getKey().getA(),e.getKey().getB());
            double abCurCount = abCount.get(pair).doubleValue();
            double cCurCount = cCount.get(e.getKey().getC()).doubleValue();

            double weight = 1.0;
            switch (vs) {
                case FIRST:
                    boxedWeight = weights.get(e.getKey().getA());
                    weight = boxedWeight == null ? 1.0 : boxedWeight;
                    break;
                case SECOND:
                    boxedWeight = weights.get(e.getKey().getB());
                    weight = boxedWeight == null ? 1.0 : boxedWeight;
                    break;
                case THIRD:
                    boxedWeight = weights.get(e.getKey().getC());
                    weight = boxedWeight == null ? 1.0 : boxedWeight;
                    break;
            }

            jmi += weight * prob * Math.log((vecLength*jointCurCount)/(abCurCount*cCurCount));
        }
        jmi /= LOG_BASE;

        double stateRatio = vecLength / jointCount.size();
        if (stateRatio < SAMPLES_RATIO) {
            logger.log(Level.INFO, "Joint MI estimate of {0} had samples/state ratio of {1}, with {2} observations and {3} states", new Object[]{jmi, stateRatio, vecLength, jointCount.size()});
        }

        return jmi;
    }

    /**
     * Calculates the discrete weighted conditional mutual information, using
     * histogram probability estimators. Arrays must be the same length.
     * @param <T1> Type contained in the first array.
     * @param <T2> Type contained in the second array.
     * @param <T3> Type contained in the condition array.
     * @param first An array of values.
     * @param second Another array of values.
     * @param condition Array to condition upon.
     * @param weights Array of weight values.
     * @return The weighted conditional mutual information I_w(first;second|condition)
     */
    public static <T1,T2,T3> double conditionalMI(List<T1> first, List<T2> second, List<T3> condition, List<Double> weights) {
        if ((first.size() == second.size()) && (first.size() == condition.size()) && (first.size() == weights.size())) {
            WeightedTripleDistribution<T1,T2,T3> tripleRV = WeightedTripleDistribution.constructFromLists(first, second, condition, weights);

            return conditionalMI(tripleRV);
        } else {
            throw new IllegalArgumentException("Weighted Conditional Mutual Information requires four vectors the same length. first.size() = " + first.size() + ", second.size() = " + second.size() + ", condition.size() = " + condition.size() + ", weights.size() = "+ weights.size());
        }
    }

    /**
     * Calculates the discrete weighted conditional mutual information, using
     * histogram probability estimators.
     * @param tripleRV The weighted triple distribution.
     * @param <T1> The first element type.
     * @param <T2> The second element type.
     * @param <T3> The condition element type.
     * @return The weighted conditional mutual information I_w(first;second|condition)
     */
    public static <T1,T2,T3> double conditionalMI(WeightedTripleDistribution<T1,T2,T3> tripleRV) {
        Map<CachedTriple<T1,T2,T3>,WeightCountTuple> jointCount = tripleRV.getJointCount();
        Map<CachedPair<T1,T3>,WeightCountTuple> acCount = tripleRV.getACCount();
        Map<CachedPair<T2,T3>,WeightCountTuple> bcCount = tripleRV.getBCCount();
        Map<T3,WeightCountTuple> cCount = tripleRV.getCCount();

        double vectorLength = tripleRV.count;
        double cmi = 0.0;
        for (Entry<CachedTriple<T1,T2,T3>,WeightCountTuple> e : jointCount.entrySet()) {
            double weight = e.getValue().weight;
            double jointCurCount = e.getValue().count;
            double prob = jointCurCount / vectorLength;
            CachedPair<T1,T3> acPair = e.getKey().getAC();
            CachedPair<T2,T3> bcPair = e.getKey().getBC();
            double acCurCount = acCount.get(acPair).count;
            double bcCurCount = bcCount.get(bcPair).count;
            double cCurCount = cCount.get(e.getKey().getC()).count;

            cmi += weight * prob * Math.log((cCurCount*jointCurCount)/(acCurCount*bcCurCount));
        }
        cmi /= LOG_BASE;

        double stateRatio = vectorLength / jointCount.size();
        if (stateRatio < SAMPLES_RATIO) {
            logger.log(Level.INFO, "Conditional MI estimate of {0} had samples/state ratio of {1}", new Object[]{cmi, stateRatio});
        }

        return cmi;
    }

    /**
     * Calculates the discrete weighted conditional mutual information, using
     * histogram probability estimators.
     * @param rv The triple distribution.
     * @param weights The element weights.
     * @param vs The variable to apply the weights to.
     * @param <T1> The first element type.
     * @param <T2> The second element type.
     * @param <T3> The condition element type.
     * @return The weighted conditional mutual information I_w(first;second|condition)
     */
    public static <T1,T2,T3> double conditionalMI(TripleDistribution<T1,T2,T3> rv, Map<?,Double> weights, VariableSelector vs) {
        Double boxedWeight;
        Map<CachedTriple<T1,T2,T3>,MutableLong> jointCount = rv.getJointCount();
        Map<CachedPair<T1,T3>,MutableLong> acCount = rv.getACCount();
        Map<CachedPair<T2,T3>,MutableLong> bcCount = rv.getBCCount();
        Map<T3,MutableLong> cCount = rv.getCCount();

        double vectorLength = rv.count;
        double cmi = 0.0;
        for (Entry<CachedTriple<T1, T2, T3>, MutableLong> e : jointCount.entrySet()) {
            double jointCurCount = e.getValue().doubleValue();
            double prob = jointCurCount / vectorLength;
            CachedPair<T1, T3> acPair = new CachedPair<>(e.getKey().getA(), e.getKey().getC());
            CachedPair<T2, T3> bcPair = new CachedPair<>(e.getKey().getB(), e.getKey().getC());
            double acCurCount = acCount.get(acPair).doubleValue();
            double bcCurCount = bcCount.get(bcPair).doubleValue();
            double cCurCount = cCount.get(e.getKey().getC()).doubleValue();

            double weight = 1.0;
            switch (vs) {
                case FIRST:
                    boxedWeight = weights.get(e.getKey().getA());
                    weight = boxedWeight == null ? 1.0 : boxedWeight;
                    break;
                case SECOND:
                    boxedWeight = weights.get(e.getKey().getB());
                    weight = boxedWeight == null ? 1.0 : boxedWeight;
                    break;
                case THIRD:
                    boxedWeight = weights.get(e.getKey().getC());
                    weight = boxedWeight == null ? 1.0 : boxedWeight;
                    break;
            }
            cmi += weight * prob * Math.log((cCurCount * jointCurCount) / (acCurCount * bcCurCount));
        }
        cmi /= LOG_BASE;

        double stateRatio = vectorLength / jointCount.size();
        if (stateRatio < SAMPLES_RATIO) {
            logger.log(Level.INFO, "Conditional MI estimate of {0} had samples/state ratio of {1}", new Object[]{cmi, stateRatio});
        }

        return cmi;
    }

    /**
     * Calculates the discrete weighted mutual information, using histogram
     * probability estimators.
     * <p>
     * Arrays must be the same length.
     * @param <T1> Type of the first array
     * @param <T2> Type of the second array
     * @param first An array of values
     * @param second Another array of values
     * @param weights Array of weight values.
     * @return The weighted mutual information I_w(first;Second)
     */
    public static <T1,T2> double mi(ArrayList<T1> first, ArrayList<T2> second, ArrayList<Double> weights) {
        if ((first.size() == second.size()) && (first.size() == weights.size())) {
            WeightedPairDistribution<T1,T2> countPair = WeightedPairDistribution.constructFromLists(first,second,weights);
            return mi(countPair);
        } else {
            throw new IllegalArgumentException("Weighted Mutual Information requires three vectors the same length. first.size() = " + first.size() + ", second.size() = " + second.size() + ", weights.size() = " + weights.size());
        }
    }

    /**
     * Calculates the discrete weighted mutual information, using histogram
     * probability estimators.
     * @param jointDist The weighted joint distribution.
     * @param <T1> Type of the first element.
     * @param <T2> Type of the second element.
     * @return The weighted mutual information I_w(first;Second)
     */
    public static <T1,T2> double mi(WeightedPairDistribution<T1,T2> jointDist) {
        double vectorLength = jointDist.count;
        double mi = 0.0;
        Map<CachedPair<T1,T2>,WeightCountTuple> countDist = jointDist.getJointCounts();
        Map<T1,WeightCountTuple> firstCountDist = jointDist.getFirstCount();
        Map<T2,WeightCountTuple> secondCountDist = jointDist.getSecondCount();

        for (Entry<CachedPair<T1,T2>,WeightCountTuple> e : countDist.entrySet()) {
            double weight = e.getValue().weight;
            double jointCount = e.getValue().count;
            double prob = jointCount / vectorLength;
            double firstCount = firstCountDist.get(e.getKey().getA()).count;
            double secondCount = secondCountDist.get(e.getKey().getB()).count;

            mi += weight * prob * Math.log((vectorLength*jointCount)/(firstCount*secondCount));
        }
        mi /= LOG_BASE;

        double stateRatio = vectorLength / countDist.size();
        if (stateRatio < SAMPLES_RATIO) {
            logger.log(Level.INFO, "MI estimate of {0} had samples/state ratio of {1}", new Object[]{mi, stateRatio});
        }

        return mi;
    }

    /**
     * Calculates the discrete weighted mutual information, using histogram
     * probability estimators.
     * @param pairDist The joint distribution.
     * @param weights The element weights.
     * @param vs The variable to apply the weights to.
     * @param <T1> Type of the first element.
     * @param <T2> Type of the second element.
     * @return The weighted mutual information I_w(first;Second)
     */
    public static <T1,T2> double mi(PairDistribution<T1,T2> pairDist, Map<?,Double> weights, VariableSelector vs) {
        if (vs == VariableSelector.THIRD) {
            throw new IllegalArgumentException("MI only has two variables");
        }
        Map<CachedPair<T1,T2>,MutableLong> countDist = pairDist.jointCounts;
        Map<T1,MutableLong> firstCountDist = pairDist.firstCount;
        Map<T2,MutableLong> secondCountDist = pairDist.secondCount;

        Double boxedWeight;
        double vectorLength = pairDist.count;
        double mi = 0.0;
        boolean error = false;
        for (Entry<CachedPair<T1,T2>,MutableLong> e : countDist.entrySet()) {
            double jointCount = e.getValue().doubleValue();
            double prob = jointCount / vectorLength;
            double firstProb = firstCountDist.get(e.getKey().getA()).doubleValue();
            double secondProb = secondCountDist.get(e.getKey().getB()).doubleValue();

            double top = vectorLength * jointCount;
            double bottom = firstProb * secondProb;
            double ratio = top/bottom;
            double logRatio = Math.log(ratio);

            if (Double.isNaN(logRatio) || Double.isNaN(prob) || Double.isNaN(mi)) {
                logger.log(Level.WARNING, "State = " + e.getKey().toString());
                logger.log(Level.WARNING, "mi = " + mi + " prob = " + prob + " top = " + top + " bottom = " + bottom + " ratio = " + ratio + " logRatio = " + logRatio);
                error = true;
            }

            double weight = 1.0;
            switch (vs) {
                case FIRST:
                    boxedWeight = weights.get(e.getKey().getA());
                    weight = boxedWeight == null ? 1.0 : boxedWeight;
                    break;
                case SECOND:
                    boxedWeight = weights.get(e.getKey().getB());
                    weight = boxedWeight == null ? 1.0 : boxedWeight;
                    break;
                default:
                    throw new IllegalArgumentException("VariableSelector.THIRD not allowed in a two variable calculation.");
            }
            mi += weight * prob * logRatio;
            //mi += prob * Math.log((vectorLength*jointCount)/(firstProb*secondProb));
        }
        mi /= LOG_BASE;

        double stateRatio = vectorLength / countDist.size();
        if (stateRatio < SAMPLES_RATIO) {
            logger.log(Level.INFO, "MI estimate of {0} had samples/state ratio of {1}", new Object[]{mi, stateRatio});
        }

        if (error) {
            logger.log(Level.SEVERE, "NanFound ", new IllegalStateException("NaN found"));
        }

        return mi;
    }

    /**
     * Calculates the Shannon/Guiasu weighted joint entropy of two arrays, 
     * using histogram probability estimators. 
     * <p>
     * Arrays must be same length.
     * @param <T1> Type of the first array.
     * @param <T2> Type of the second array.
     * @param first An array of values.
     * @param second Another array of values.
     * @param weights Array of weight values.
     * @return The entropy H(first,second)
     */
    public static <T1,T2> double jointEntropy(ArrayList<T1> first, ArrayList<T2> second, ArrayList<Double> weights) {
        if ((first.size() == second.size()) && (first.size() == weights.size())) {
            double vectorLength = first.size();
            double jointEntropy = 0.0;
            
            WeightedPairDistribution<T1,T2> pairDist = WeightedPairDistribution.constructFromLists(first,second,weights);
            Map<CachedPair<T1,T2>,WeightCountTuple> countDist = pairDist.getJointCounts();

            for (Entry<CachedPair<T1,T2>,WeightCountTuple> e : countDist.entrySet()) {
                double prob = e.getValue().count / vectorLength;
                double weight = e.getValue().weight;

                jointEntropy -= weight * prob * Math.log(prob);
            }
            jointEntropy /= LOG_BASE;

            double stateRatio = vectorLength / countDist.size();
            if (stateRatio < SAMPLES_RATIO) {
                logger.log(Level.INFO, "Weighted Joint Entropy estimate of {0} had samples/state ratio of {1}", new Object[]{jointEntropy, stateRatio});
            }
            
            return jointEntropy;
        } else {
            throw new IllegalArgumentException("Weighted Joint Entropy requires three vectors the same length. first.size() = " + first.size() + ", second.size() = " + second.size() + ", weights.size() = " + weights.size());
        }
    }
    
    /**
     * Calculates the discrete Shannon/Guiasu Weighted Conditional Entropy of 
     * two arrays, using histogram probability estimators. 
     * <p>
     * Arrays must be the same length.
     * @param <T1> Type of the first array.
     * @param <T2> Type of the second array.
     * @param vector The main array of values.
     * @param condition The array to condition on.
     * @param weights Array of weight values.
     * @return The weighted conditional entropy H_w(vector|condition).
     */
    public static <T1,T2> double weightedConditionalEntropy(ArrayList<T1> vector, ArrayList<T2> condition, ArrayList<Double> weights) {
        if ((vector.size() == condition.size()) && (vector.size() == weights.size())) {
            double vectorLength = vector.size();
            double condEntropy = 0.0;
            
            WeightedPairDistribution<T1,T2> pairDist = WeightedPairDistribution.constructFromLists(vector,condition,weights);
            Map<CachedPair<T1,T2>,WeightCountTuple> countDist = pairDist.getJointCounts();
            Map<T2,WeightCountTuple> conditionCountDist = pairDist.getSecondCount();

            for (Entry<CachedPair<T1,T2>,WeightCountTuple> e : countDist.entrySet()) {
                double prob = e.getValue().count / vectorLength;
                double condProb = conditionCountDist.get(e.getKey().getB()).count / vectorLength;
                double weight = e.getValue().weight;

                condEntropy -= weight * prob * Math.log(prob/condProb);
            }
            condEntropy /= LOG_BASE;

            double stateRatio = vectorLength / countDist.size();
            if (stateRatio < SAMPLES_RATIO) {
                logger.log(Level.INFO, "Weighted Conditional Entropy estimate of {0} had samples/state ratio of {1}", new Object[]{condEntropy, stateRatio});
            }
            
            return condEntropy;
        } else {
            throw new IllegalArgumentException("Weighted Conditional Entropy requires three vectors the same length. vector.size() = " + vector.size() + ", condition.size() = " + condition.size() + ", weights.size() = " + weights.size());
        }
    }

    /**
     * Calculates the discrete Shannon/Guiasu Weighted Entropy, using histogram 
     * probability estimators.
     * @param <T> Type of the array.
     * @param vector The array of values.
     * @param weights Array of weight values.
     * @return The weighted entropy H_w(vector).
     */
    public static <T> double weightedEntropy(ArrayList<T> vector, ArrayList<Double> weights) {
        if (vector.size() == weights.size()) {
            double vectorLength = vector.size();
            double entropy = 0.0;

            Map<T,WeightCountTuple> countDist = calculateWeightedCountDist(vector,weights);
            for (Entry<T,WeightCountTuple> e : countDist.entrySet()) {
                long count = e.getValue().count;
                double weight = e.getValue().weight;
                double prob = count / vectorLength;
                entropy -= weight * prob * Math.log(prob);
            }
            entropy /= LOG_BASE;

            double stateRatio = vectorLength / countDist.size();
            if (stateRatio < SAMPLES_RATIO) {
                logger.log(Level.INFO, "Weighted Entropy estimate of {0} had samples/state ratio of {1}", new Object[]{entropy, stateRatio});
            }
            
            return entropy;
        } else {
            throw new IllegalArgumentException("Weighted Entropy requires two vectors the same length. vector.size() = " + vector.size() + ",weights.size() = " + weights.size());
        }
    }

    /**
     * Generate the counts for a single vector.
     * @param <T> The type inside the vector.
     * @param vector An array of values.
     * @param weights The array of weight values.
     * @return A HashMap from states of T to Pairs of count and total weight for that state.
     */
    public static <T> Map<T,WeightCountTuple> calculateWeightedCountDist(ArrayList<T> vector, ArrayList<Double> weights) {
        Map<T,WeightCountTuple> dist = new LinkedHashMap<>(DEFAULT_MAP_SIZE);
        for (int i = 0; i < vector.size(); i++) {
            T e = vector.get(i);
            Double weight = weights.get(i);
            WeightCountTuple curVal = dist.computeIfAbsent(e,(k) -> new WeightCountTuple());
            curVal.count += 1;
            curVal.weight += weight;
        }

        normaliseWeights(dist);

        return dist;
    }

    /**
     * Normalizes the weights in the map, i.e., divides each weight by it's count.
     * @param map The map to normalize.
     * @param <T> The type of the variable that was counted.
     */
    public static <T> void normaliseWeights(Map<T,WeightCountTuple> map) {
        for (Entry<T,WeightCountTuple> e : map.entrySet()) {
            WeightCountTuple tuple = e.getValue();
            tuple.weight /= tuple.count;
        }
    }
    
}
