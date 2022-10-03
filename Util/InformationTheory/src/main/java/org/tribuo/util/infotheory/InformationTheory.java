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

package org.tribuo.util.infotheory;

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.util.infotheory.impl.CachedPair;
import org.tribuo.util.infotheory.impl.CachedTriple;
import org.tribuo.util.infotheory.impl.PairDistribution;
import org.tribuo.util.infotheory.impl.Row;
import org.tribuo.util.infotheory.impl.RowList;
import org.tribuo.util.infotheory.impl.TripleDistribution;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * A class of (discrete) information theoretic functions. Gives warnings if
 * there are insufficient samples to estimate the quantities accurately.
 * <p>
 * Defaults to log_2, so returns values in bits.
 * <p>
 * All functions expect that the element types have well defined equals and
 * hashcode, and that equals is consistent with hashcode. The behaviour is undefined
 * if this is not true.
 */
public final class InformationTheory {
    private static final Logger logger = Logger.getLogger(InformationTheory.class.getName());

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
     * Private constructor, only has static methods.
     */
    private InformationTheory() {}

    /**
     * Calculates the mutual information between the two sets of random variables.
     * @param first The first set of random variables.
     * @param second The second set of random variables.
     * @param <T1> The first type.
     * @param <T2> The second type.
     * @return The mutual information I(first;second).
     */
    public static <T1,T2> double mi(Set<List<T1>> first, Set<List<T2>> second) {
        List<Row<T1>> firstList = new RowList<>(first);
        List<Row<T2>> secondList = new RowList<>(second);

        return mi(firstList,secondList);
    }

    /**
     * Calculates the conditional mutual information between first and second conditioned on the set.
     * @param first A sample from the first random variable.
     * @param second A sample from the second random variable.
     * @param condition A sample from the conditioning set of random variables.
     * @param <T1> The first type.
     * @param <T2> The second type.
     * @param <T3> The third type.
     * @return The conditional mutual information I(first;second|condition).
     */
    public static <T1,T2,T3> double cmi(List<T1> first, List<T2> second, Set<List<T3>> condition) {
        if (condition.isEmpty()) {
            //logger.log(Level.INFO,"Empty conditioning set");
            return mi(first,second);
        } else {
            List<Row<T3>> conditionList = new RowList<>(condition);
        
            return conditionalMI(first,second,conditionList);
        }
    }

    /**
     * Calculates the GTest statistics for the input variables conditioned on the set.
     * @param first A sample from the first random variable.
     * @param second A sample from the second random variable.
     * @param condition A sample from the conditioning set of random variables.
     * @param <T1> The first type.
     * @param <T2> The second type.
     * @param <T3> The third type.
     * @return The GTest statistics.
     */
    public static <T1,T2,T3> GTestStatistics gTest(List<T1> first, List<T2> second, Set<List<T3>> condition) {
        ScoreStateCountTuple tuple;
        if (condition == null) {
            //logger.log(Level.INFO,"Null conditioning set");
            tuple = innerMI(first,second);
        } else if (condition.isEmpty()) {
            //logger.log(Level.INFO,"Empty conditioning set");
            tuple = innerMI(first,second);
        } else {
            List<Row<T3>> conditionList = new RowList<>(condition);
        
            tuple = innerConditionalMI(first,second,conditionList);
        }
        double gMetric = 2 * second.size() * tuple.score;
        double prob = computeChiSquaredProbability(tuple.stateCount, gMetric);
        return new GTestStatistics(gMetric,tuple.stateCount,prob);
    }

    /**
     * Computes the cumulative probability of the input value under a Chi-Squared distribution
     * with the specified degrees of Freedom.
     * @param degreesOfFreedom The degrees of freedom in the distribution.
     * @param value The observed value.
     * @return The cumulative probability of the observed value.
     */
    private static double computeChiSquaredProbability(int degreesOfFreedom, double value) {
        if (value <= 0) {
            return 0.0;
        } else {
            int shape = degreesOfFreedom / 2;
            int scale = 2;
            return Gamma.regularizedGammaP(shape, value / scale, 1e-14, Integer.MAX_VALUE);
        }
    }

    /**
     * Calculates the discrete Shannon joint mutual information, using
     * histogram probability estimators. Arrays must be the same length.
     * @param <T1> Type contained in the first array.
     * @param <T2> Type contained in the second array.
     * @param <T3> Type contained in the target array.
     * @param first An array of values.
     * @param second Another array of values.
     * @param target Target array of values.
     * @return The mutual information I(first,second;joint)
     */
    public static <T1,T2,T3> double jointMI(List<T1> first, List<T2> second, List<T3> target) {
        if ((first.size() == second.size()) && (first.size() == target.size())) {
            TripleDistribution<T1,T2,T3> tripleRV = TripleDistribution.constructFromLists(first,second,target);
            return jointMI(tripleRV);
        } else {
            throw new IllegalArgumentException("Joint Mutual Information requires three vectors the same length. first.size() = " + first.size() + ", second.size() = " + second.size() + ", target.size() = " + target.size());
        }
    }

    /**
     * Calculates the discrete Shannon joint mutual information, using
     * histogram probability estimators. Arrays must be the same length.
     * @param <T1> Type contained in the first array.
     * @param <T2> Type contained in the second array.
     * @param <T3> Type contained in the target array.
     * @param rv The random variable to calculate the joint mi of
     * @return The mutual information I(first,second;joint)
     */
    public static <T1,T2,T3> double jointMI(TripleDistribution<T1,T2,T3> rv) {
        double vecLength = rv.count;
        Map<CachedTriple<T1,T2,T3>,MutableLong> jointCount = rv.getJointCount();
        Map<CachedPair<T1,T2>,MutableLong> abCount = rv.getABCount();
        Map<T3,MutableLong> cCount = rv.getCCount();

        double jmi = 0.0;
        for (Entry<CachedTriple<T1,T2,T3>,MutableLong> e : jointCount.entrySet()) {
            double jointCurCount = e.getValue().doubleValue();
            double prob = jointCurCount / vecLength;
            CachedPair<T1,T2> pair = e.getKey().getAB();
            double abCurCount = abCount.get(pair).doubleValue();
            double cCurCount = cCount.get(e.getKey().getC()).doubleValue();

            jmi += prob * Math.log((vecLength*jointCurCount)/(abCurCount*cCurCount));
        }
        jmi /= LOG_BASE;
        
        double stateRatio = vecLength / jointCount.size();
        if (stateRatio < SAMPLES_RATIO) {
            logger.log(Level.INFO, "Joint MI estimate of {0} had samples/state ratio of {1}, with {2} observations and {3} states", new Object[]{jmi, stateRatio, vecLength, jointCount.size()});
        }

        return jmi;
    }

    /**
     * Calculates the conditional mutual information. If flipped == true, then calculates I(T1;T3|T2), otherwise calculates I(T1;T2|T3).
     * @param <T1> The type of the first argument.
     * @param <T2> The type of the second argument.
     * @param <T3> The type of the third argument.
     * @param rv The random variable.
     * @param flipped If true then the second element is the conditional variable, otherwise the third element is.
     * @return A ScoreStateCountTuple containing the conditional mutual information and the number of states in the joint random variable.
     */
    private static <T1,T2,T3> ScoreStateCountTuple innerConditionalMI(TripleDistribution<T1,T2,T3> rv, boolean flipped) {
        Map<CachedTriple<T1,T2,T3>,MutableLong> jointCount = rv.getJointCount();
        Map<CachedPair<T1,T2>,MutableLong> abCount = rv.getABCount();
        Map<CachedPair<T1,T3>,MutableLong> acCount = rv.getACCount();
        Map<CachedPair<T2,T3>,MutableLong> bcCount = rv.getBCCount();
        Map<T2,MutableLong> bCount = rv.getBCount();
        Map<T3,MutableLong> cCount = rv.getCCount();

        double vectorLength = rv.count;
        double cmi = 0.0;
        if (flipped) {
            for (Entry<CachedTriple<T1,T2,T3>, MutableLong> e : jointCount.entrySet()) {
                double jointCurCount = e.getValue().doubleValue();
                double prob = jointCurCount / vectorLength;
                CachedPair<T1,T2> abPair = e.getKey().getAB();
                CachedPair<T2,T3> bcPair = e.getKey().getBC();
                double abCurCount = abCount.get(abPair).doubleValue();
                double bcCurCount = bcCount.get(bcPair).doubleValue();
                double bCurCount = bCount.get(e.getKey().getB()).doubleValue();

                cmi += prob * Math.log((bCurCount * jointCurCount) / (abCurCount * bcCurCount));
            }
        } else {
            for (Entry<CachedTriple<T1, T2, T3>, MutableLong> e : jointCount.entrySet()) {
                double jointCurCount = e.getValue().doubleValue();
                double prob = jointCurCount / vectorLength;
                CachedPair<T1, T3> acPair = e.getKey().getAC();
                CachedPair<T2, T3> bcPair = e.getKey().getBC();
                double acCurCount = acCount.get(acPair).doubleValue();
                double bcCurCount = bcCount.get(bcPair).doubleValue();
                double cCurCount = cCount.get(e.getKey().getC()).doubleValue();

                cmi += prob * Math.log((cCurCount * jointCurCount) / (acCurCount * bcCurCount));
            }
        }
        cmi /= LOG_BASE;

        double stateRatio = vectorLength / jointCount.size();
        if (stateRatio < SAMPLES_RATIO) {
            logger.log(Level.INFO, "Conditional MI estimate of {0} had samples/state ratio of {1}", new Object[]{cmi, stateRatio});
        }
        
        return new ScoreStateCountTuple(cmi,jointCount.size());
    }

    /**
     * Calculates the conditional mutual information, I(T1;T2|T3).
     * @param <T1> The type of the first argument.
     * @param <T2> The type of the second argument.
     * @param <T3> The type of the third argument.
     * @param first The first random variable.
     * @param second The second random variable.
     * @param condition The conditioning random variable.
     * @return A ScoreStateCountTuple containing the conditional mutual information and the number of states in the joint random variable.
     */
    private static <T1,T2,T3> ScoreStateCountTuple innerConditionalMI(List<T1> first, List<T2> second, List<T3> condition) {
        if ((first.size() == second.size()) && (first.size() == condition.size())) {
            TripleDistribution<T1,T2,T3> tripleRV = TripleDistribution.constructFromLists(first,second,condition);

            return innerConditionalMI(tripleRV,false);
        } else {
            throw new IllegalArgumentException("Conditional Mutual Information requires three vectors the same length. first.size() = " + first.size() + ", second.size() = " + second.size() + ", condition.size() = " + condition.size());
        }
    }
    
    /**
     * Calculates the discrete Shannon conditional mutual information, using
     * histogram probability estimators. Arrays must be the same length.
     * @param <T1> Type contained in the first array.
     * @param <T2> Type contained in the second array.
     * @param <T3> Type contained in the condition array.
     * @param first An array of values.
     * @param second Another array of values.
     * @param condition Array to condition upon.
     * @return The conditional mutual information I(first;second|condition)
     */
    public static <T1,T2,T3> double conditionalMI(List<T1> first, List<T2> second, List<T3> condition) {
        return innerConditionalMI(first,second,condition).score;
    }

    /**
     * Calculates the discrete Shannon conditional mutual information, using
     * histogram probability estimators. Note this calculates I(T1;T2|T3).
     * @param <T1> Type of the first variable.
     * @param <T2> Type of the second variable.
     * @param <T3> Type of the condition variable.
     * @param rv The triple random variable of the three inputs.
     * @return The conditional mutual information I(first;second|condition)
     */
    public static <T1,T2,T3> double conditionalMI(TripleDistribution<T1,T2,T3> rv) {
        return innerConditionalMI(rv,false).score;
    }

    /**
     * Calculates the discrete Shannon conditional mutual information, using
     * histogram probability estimators. Note this calculates I(T1;T3|T2).
     * @param <T1> Type of the first variable.
     * @param <T2> Type of the condition variable.
     * @param <T3> Type of the second variable.
     * @param rv The triple random variable of the three inputs.
     * @return The conditional mutual information I(first;second|condition)
     */
    public static <T1,T2,T3> double conditionalMIFlipped(TripleDistribution<T1,T2,T3> rv) {
        return innerConditionalMI(rv,true).score;
    }

    /**
     * Calculates the mutual information from a joint random variable.
     * @param pairDist The joint distribution.
     * @param <T1> The first type.
     * @param <T2> The second type.
     * @return A ScoreStateCountTuple containing the mutual information and the number of states in the joint variable.
     */
    private static <T1,T2> ScoreStateCountTuple innerMI(PairDistribution<T1,T2> pairDist) {
        Map<CachedPair<T1,T2>,MutableLong> countDist = pairDist.jointCounts;
        Map<T1,MutableLong> firstCountDist = pairDist.firstCount;
        Map<T2,MutableLong> secondCountDist = pairDist.secondCount;

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
            mi += prob * logRatio;
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
        
        return new ScoreStateCountTuple(mi,countDist.size());
    }

    /**
     * Calculates the mutual information between the two lists.
     * @param first The first list.
     * @param second The second list.
     * @param <T1> The first type.
     * @param <T2> The second type.
     * @return A ScoreStateCountTuple containing the mutual information and the number of states in the joint variable.
     */
    private static <T1,T2> ScoreStateCountTuple innerMI(List<T1> first, List<T2> second) {
        if (first.size() == second.size()) {
            PairDistribution<T1,T2> pairDist = PairDistribution.constructFromLists(first, second);
            
            return innerMI(pairDist);
        } else {
            throw new IllegalArgumentException("Mutual Information requires two vectors the same length. first.size() = " + first.size() + ", second.size() = " + second.size());
        }
    }
    
    /**
     * Calculates the discrete Shannon mutual information, using histogram 
     * probability estimators. Arrays must be the same length.
     * @param <T1> Type of the first array
     * @param <T2> Type of the second array
     * @param first An array of values
     * @param second Another array of values
     * @return The mutual information I(first;second)
     */
    public static <T1,T2> double mi(List<T1> first, List<T2> second) {
        return innerMI(first,second).score;
    }

    /**
     * Calculates the discrete Shannon mutual information, using histogram 
     * probability estimators.
     * @param <T1> Type of the first variable
     * @param <T2> Type of the second variable
     * @param pairDist PairDistribution for the two variables.
     * @return The mutual information I(first;second)
     */
    public static <T1,T2> double mi(PairDistribution<T1,T2> pairDist) {
        return innerMI(pairDist).score;
    }

    /**
     * Calculates the Shannon joint entropy of two arrays, using histogram 
     * probability estimators. Arrays must be same length.
     * @param <T1> Type of the first array.
     * @param <T2> Type of the second array.
     * @param first An array of values.
     * @param second Another array of values.
     * @return The entropy H(first,second)
     */
    public static <T1,T2> double jointEntropy(List<T1> first, List<T2> second) {
        if (first.size() == second.size()) {
            double vectorLength = first.size();
            double jointEntropy = 0.0;
            
            PairDistribution<T1,T2> countPair = PairDistribution.constructFromLists(first,second); 
            Map<CachedPair<T1,T2>,MutableLong> countDist = countPair.jointCounts;

            for (Entry<CachedPair<T1,T2>,MutableLong> e : countDist.entrySet()) {
                double prob = e.getValue().doubleValue() / vectorLength;

                jointEntropy -= prob * Math.log(prob);
            }
            jointEntropy /= LOG_BASE;

            double stateRatio = vectorLength / countDist.size();
            if (stateRatio < SAMPLES_RATIO) {
                logger.log(Level.INFO, "Joint Entropy estimate of {0} had samples/state ratio of {1}", new Object[]{jointEntropy, stateRatio});
            }
            
            return jointEntropy;
        } else {
            throw new IllegalArgumentException("Joint Entropy requires two vectors the same length. first.size() = " + first.size() + ", second.size() = " + second.size());
        }
    }
    
    /**
     * Calculates the discrete Shannon conditional entropy of two arrays, using
     * histogram probability estimators. Arrays must be the same length.
     * @param <T1> Type of the first array.
     * @param <T2> Type of the second array.
     * @param vector The main array of values.
     * @param condition The array to condition on.
     * @return The conditional entropy H(vector|condition).
     */
    public static <T1,T2> double conditionalEntropy(List<T1> vector, List<T2> condition) {
        if (vector.size() == condition.size()) {
            double vectorLength = vector.size();
            double condEntropy = 0.0;
            
            PairDistribution<T1,T2> countPair = PairDistribution.constructFromLists(vector,condition); 
            Map<CachedPair<T1,T2>,MutableLong> countDist = countPair.jointCounts;
            Map<T2,MutableLong> conditionCountDist = countPair.secondCount;

            for (Entry<CachedPair<T1,T2>,MutableLong> e : countDist.entrySet()) {
                double prob = e.getValue().doubleValue() / vectorLength;
                double condProb = conditionCountDist.get(e.getKey().getB()).doubleValue() / vectorLength;

                condEntropy -= prob * Math.log(prob/condProb);
            }
            condEntropy /= LOG_BASE;

            double stateRatio = vectorLength / countDist.size();
            if (stateRatio < SAMPLES_RATIO) {
                logger.log(Level.INFO, "Conditional Entropy estimate of {0} had samples/state ratio of {1}", new Object[]{condEntropy, stateRatio});
            }
            
            return condEntropy;
        } else {
            throw new IllegalArgumentException("Conditional Entropy requires two vectors the same length. vector.size() = " + vector.size() + ", condition.size() = " + condition.size());
        }
    }

    /**
     * Calculates the discrete Shannon entropy, using histogram probability 
     * estimators.
     * @param <T> Type of the array.
     * @param vector The array of values.
     * @return The entropy H(vector).
     */
    public static <T> double entropy(List<T> vector) {
        double vectorLength = vector.size();
        double entropy = 0.0;

        Map<T,Long> countDist = calculateCountDist(vector);
        for (Entry<T,Long> e : countDist.entrySet()) {
            double prob = e.getValue() / vectorLength;
            entropy -= prob * Math.log(prob);
        }
        entropy /= LOG_BASE;

        double stateRatio = vectorLength / countDist.size();
        if (stateRatio < SAMPLES_RATIO) {
            logger.log(Level.INFO, "Entropy estimate of {0} had samples/state ratio of {1}", new Object[]{entropy, stateRatio});
        }
        
        return entropy;
    }

    /**
     * Generate the counts for a single vector.
     * @param <T> The type inside the vector.
     * @param vector An array of values.
     * @return A HashMap from states of T to counts.
     */
    public static <T> Map<T,Long> calculateCountDist(List<T> vector) {
        HashMap<T,Long> countDist = new HashMap<>(DEFAULT_MAP_SIZE);
        for (T e : vector) {
            Long curCount = countDist.getOrDefault(e,0L);
            curCount += 1;
            countDist.put(e, curCount);
        }

        return countDist;
    }

    /**
     * Calculates the discrete Shannon entropy of a stream, assuming each element of the stream is
     * an element of the same probability distribution.
     * @param vector The probability distribution.
     * @return The entropy.
     */
    public static double calculateEntropy(Stream<Double> vector) {
        return vector.map((p) -> (- p * Math.log(p) / LOG_BASE)).reduce(0.0, Double::sum);
    }

    /**
     * Calculates the discrete Shannon entropy of a stream, assuming each element of the stream is
     * an element of the same probability distribution.
     * @param vector The probability distribution.
     * @return The entropy.
     */
    public static double calculateEntropy(DoubleStream vector) {
        return vector.map((p) -> (- p * Math.log(p) / LOG_BASE)).sum();
    }

    /**
     * Compute the expected mutual information assuming randomized inputs.
     *
     * @param first The first vector.
     * @param second The second vector.
     * @param <T> The type inside the list. Must define equals and hashcode.
     * @return The expected mutual information under a hypergeometric distribution.
     */
    public static <T> double expectedMI(List<T> first, List<T> second) {
        PairDistribution<T,T> pd = PairDistribution.constructFromLists(first,second);

        Map<T, MutableLong> firstCount = pd.firstCount;
        Map<T,MutableLong> secondCount = pd.secondCount;
        long count = pd.count;

        double output = 0.0;

        for (Entry<T,MutableLong> f : firstCount.entrySet()) {
            for (Entry<T,MutableLong> s : secondCount.entrySet()) {
                long fVal = f.getValue().longValue();
                long sVal = s.getValue().longValue();
                long minCount = Math.min(fVal, sVal);

                long threshold = fVal + sVal - count;
                long start = threshold > 1 ? threshold : 1;

                for (long nij = start; nij <= minCount; nij++) {
                    double acc = ((double) nij) / count;
                    acc *= Math.log(((double) (count * nij)) / (fVal * sVal));
                    //numerator
                    double logSpace = Gamma.logGamma(fVal + 1);
                    logSpace += Gamma.logGamma(sVal + 1);
                    logSpace += Gamma.logGamma(count - fVal + 1);
                    logSpace += Gamma.logGamma(count - sVal + 1);
                    //denominator
                    logSpace -= Gamma.logGamma(count + 1);
                    logSpace -= Gamma.logGamma(nij + 1);
                    logSpace -= Gamma.logGamma(fVal - nij + 1);
                    logSpace -= Gamma.logGamma(sVal - nij + 1);
                    logSpace -= Gamma.logGamma(count - fVal - sVal + nij + 1);
                    acc *= Math.exp(logSpace);
                    output += acc;
                }
            }
        }
        return output;
    }

    /**
     * A tuple of the information theoretic value, along with the number of
     * states in the random variable. Will be a record one day.
     */
    private static class ScoreStateCountTuple {
        public final double score;
        public final int stateCount;

        /**
         * Construct a score state tuple
         * @param score The score.
         * @param stateCount The number of states.
         */
        ScoreStateCountTuple(double score, int stateCount) {
            this.score = score;
            this.stateCount = stateCount;
        }

        @Override
        public String toString() {
            return "ScoreStateCount(score=" + score + ",stateCount=" + stateCount + ")";
        }
    }

    /**
     * An immutable named tuple containing the statistics from a G test.
     * <p>
     * Will be a record one day.
     */
    public static final class GTestStatistics {
        /**
         * The G test statistic.
         */
        public final double gStatistic;
        /**
         * The number of states.
         */
        public final int numStates;
        /**
         * The probability of that statistic.
         */
        public final double probability;

        /**
         * Constructs a GTestStatistics tuple with the supplied values.
         * @param gStatistic The g test statistic.
         * @param numStates The number of states.
         * @param probability The probability of that statistic.
         */
        // TODO should be package private.
        public GTestStatistics(double gStatistic, int numStates, double probability) {
            this.gStatistic = gStatistic;
            this.numStates = numStates;
            this.probability = probability;
        }

        @Override
        public String toString() {
            return "GTest(statistic="+gStatistic+",probability="+probability+",numStates="+numStates+")";
        }
    }
}

