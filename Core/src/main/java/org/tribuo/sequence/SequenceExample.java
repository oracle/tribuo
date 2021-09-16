/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.sequence;

import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.FeatureMap;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.hash.HashedFeatureMap;
import org.tribuo.impl.ArrayExample;
import org.tribuo.impl.BinaryFeaturesExample;
import org.tribuo.util.Merger;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Logger;

/**
 * A sequence of examples, used for sequence classification.
 */
public class SequenceExample<T extends Output<T>> implements Iterable<Example<T>>, Serializable {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(SequenceExample.class.getName());

    /**
     * The default sequence example weight.
     */
    public static final float DEFAULT_WEIGHT = 1.0f;

    private final List<Example<T>> examples;
    private float weight = 1.0f;

    /**
     * Creates an empty sequence example.
     */
    public SequenceExample() {
        this(new ArrayList<>());
    }

    /**
     * Creates a sequence example from the list of examples.
     * <p>
     * The examples are not copied by this method.
     * @param examples The examples to incorporate.
     */
    public SequenceExample(List<Example<T>> examples) {
        this(examples,DEFAULT_WEIGHT);
    }

    /**
     * Creates a sequence example from the list of examples, setting the weight.
     * <p>
     * The examples are encapsulated by this constructor, not copied.
     * @param examples The examples to incorporate.
     * @param weight The weight of this sequence.
     */
    public SequenceExample(List<Example<T>> examples, float weight) {
        this.examples = examples;
        this.weight = weight;
    }

    /**
     * Creates a sequence example from the supplied outputs and list of list of features.
     * <p>
     * The features are copied out by this constructor. The outputs and features lists
     * must be of the same length. Sets the weight to {@link SequenceExample#DEFAULT_WEIGHT}.
     * @param outputs The outputs for each sequence element.
     * @param features The features for each sequence element.
     */
    public SequenceExample(List<T> outputs, List<? extends List<? extends Feature>> features) {
        this(outputs,features,DEFAULT_WEIGHT);
    }

    /**
     * Creates a sequence example from the supplied weight, outputs and list of list of features.
     * <p>
     * The features are copied out by this constructor. The outputs and features lists
     * must be of the same length.
     * @param outputs The outputs for each sequence element.
     * @param features The features for each sequence element.
     * @param weight The weight for this sequence example.
     */
    public SequenceExample(List<T> outputs, List<? extends List<? extends Feature>> features, float weight) {
        this(outputs, features, weight, false);
    }

    /**
     * Creates a sequence example from the supplied weight, outputs and list of list of features.
     * <p>
     * The features are copied out by this constructor. The outputs and features lists
     * must be of the same length. Sets the weight to {@link SequenceExample#DEFAULT_WEIGHT}.
     * @param outputs The outputs for each sequence element.
     * @param features The features for each sequence element.
     * @param attemptBinaryFeatures Attempt to use {@link BinaryFeaturesExample} as the inner examples.
     */
    public SequenceExample(List<T> outputs, List<? extends List<? extends Feature>> features, boolean attemptBinaryFeatures) {
        this(outputs, features, DEFAULT_WEIGHT, attemptBinaryFeatures);
    }

    /**
     * Creates a sequence example from the supplied weight, outputs and list of list of features.
     * <p>
     * The features are copied out by this constructor. The outputs and features lists
     * must be of the same length.
     * @param outputs The outputs for each sequence element.
     * @param features The features for each sequence element.
     * @param weight The weight for this sequence example.
     * @param attemptBinaryFeatures Attempt to use {@link BinaryFeaturesExample} as the inner examples.
     */
    public SequenceExample(List<T> outputs, List<? extends List<? extends Feature>> features, float weight, boolean attemptBinaryFeatures) {
        if (outputs.size() != features.size()) {
            throw new IllegalArgumentException("outputs.size() = " + outputs.size() + ", features.size() = " + features.size());
        }

        List<Example<T>> examples = new ArrayList<>(outputs.size());

        for (int i = 0; i < outputs.size(); i++) {
            List<? extends Feature> list = features.get(i);
            Example<T> example = null;
            if(attemptBinaryFeatures){
                try {
                    example = new BinaryFeaturesExample<>(outputs.get(i), list);
                } catch(IllegalArgumentException iae){
                    logger.finer("attempted to create BinaryFeaturesExample but not all of the features were binary");
                    example = new ArrayExample<>(outputs.get(i), list);
                }
            } else {
                example = new ArrayExample<>(outputs.get(i), list);
            }
            examples.add(example);
        }

        this.examples = examples;
        this.weight = weight;
    }

    /**
     * Creates a deep copy of the supplied sequence example.
     * @param other The sequence example to copy.
     */
    public SequenceExample(SequenceExample<T> other) {
        this.examples = new ArrayList<>(other.size());
        for(Example<T> example : other) {
            examples.add(example.copy());
        }
        this.weight = other.weight;
    }

    /**
     * Return how many examples are in this sequence.
     * @return The number of examples.
     */
    public int size() {
        return examples.size();
    }

    /**
     * Removes the features in the supplied list from each example contained in this sequence.
     * @param features The features to remove.
     */
    public void removeFeatures(List<Feature> features) {
        for (Example<T> e : examples) {
            e.removeFeatures(features);
        }
    }

    /**
     * Gets the example found at the specified index.
     * @param i The index to lookup.
     * @return The {@link Example} for index i.
     */
    public Example<T> get(int i) {
        return examples.get(i);
    }

    /**
     * Checks that each {@link Example} in this sequence is valid.
     * @return True if each {@link Example} is valid, false otherwise.
     */
    public boolean validateExample() {
        if (examples.isEmpty()) {
            return false;
        } else {
            boolean valid = true;
            for (Example<T> e : examples) {
                valid &= e.validateExample();
            }
            return valid;
        }
    }

    /**
     * Reduces the features in each example using the supplied {@link Merger}.
     * @param merger The merger to use in the reduction.
     */
    public void reduceByName(Merger merger) {
        for (Example<T> e : examples) {
            e.reduceByName(merger);
        }
    }

    /**
     * Sets the weight of this sequence.
     * @param weight The new weight.
     */
    public void setWeight(float weight) {
        this.weight = weight;
    }

    /**
     * Gets the weight of this sequence.
     * @return The weight of this sequence.
     */
    public float getWeight() {
        return weight;
    }

    /**
     * Adds an {@link Example} to this sequence.
     * @param e The example to add.
     */
    public void addExample(Example<T> e) {
        examples.add(e);
    }

    /**
     * Returns a deep copy of this SequenceExample.
     * @return A deep copy.
     */
    public SequenceExample<T> copy() {
        return new SequenceExample<>(this);
    }

    @Override
    public Iterator<Example<T>> iterator() {
        return examples.iterator();
    }

    /**
     * Creates an iterator over every feature in this sequence.
     * @return An iterator over features.
     */
    public Iterator<Feature> featureIterator() {
        return new FeatureIterator<>(iterator());
    }

    /**
     * Is this sequence example dense wrt the supplied feature map.
     * <p>
     * A sequence example is "dense" if each example inside it contains all the features in the map,
     * and only those features.
     * @param fMap The feature map to check against.
     * @return True if this sequence example contains only the features in the map, and all the features in the map.
     */
    public boolean isDense(FeatureMap fMap) {
        for (Example<T> e : examples) {
            if (!e.isDense(fMap)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Converts all implicit zeros into explicit zeros based on the supplied feature map.
     * @param fMap The feature map to use for densification.
     */
    public void densify(FeatureMap fMap) {
        // Densify! - guitar solo
        for (Example<T> e : examples) {
            e.densify(fMap);
        }
    }

    /**
     * Reassigns feature name Strings in each Example inside this SequenceExample to point to
     * those in the {@link FeatureMap}. This significantly reduces memory allocation. It is called
     * when a SequenceExample is added to a {@link MutableSequenceDataset}, and should not be
     * called outside of that context as it may interact unexpectedly with
     * {@link HashedFeatureMap}.
     * @param featureMap The feature map containing canonical feature names.
     */
    public void canonicalise(FeatureMap featureMap) {
        for (Example<T> e : examples) {
            e.canonicalize(featureMap);
        }
    }

    /**
     * Creates a SequenceExample using {@link OutputFactory#getUnknownOutput()} as the output for each
     * sequence element.
     * <p>
     * Note: this method is used to create SequenceExamples at prediction time where there is no
     * ground truth {@link Output}.
     * @param features The features for each sequence element.
     * @param outputFactory The output factory to use.
     * @param <T> The type of the {@link Output}.
     * @return A new SequenceExample.
     */
    public static <T extends Output<T>> SequenceExample<T> createWithEmptyOutputs(List<? extends List<? extends Feature>> features, OutputFactory<T> outputFactory) {
        ArrayList<Example<T>> examples = new ArrayList<>(features.size());

        for (List<? extends Feature> list : features) {
            ArrayExample<T> example = new ArrayExample<>(outputFactory.getUnknownOutput());
            example.addAll(list);
            examples.add(example);
        }

        return new SequenceExample<>(examples);
    }

    private static class FeatureIterator<T extends Output<T>> implements Iterator<Feature> {
        private final Iterator<Example<T>> itr;
        private Iterator<Feature> featureItr;

        public FeatureIterator(Iterator<Example<T>> e) {
            itr = e;
        }

        @Override
        public boolean hasNext() {
            if ((featureItr != null) && (featureItr.hasNext())) {
                return true;
            } else if (itr.hasNext()) {
                while (itr.hasNext()) {
                    featureItr = itr.next().iterator();
                    if (featureItr.hasNext()) {
                        return true;
                    }
                }
                return false;
            } else {
                return false;
            }
        }

        @Override
        public Feature next() {
            if (featureItr != null) {
                return featureItr.next();
            } else {
                return null;
            }
        }
    }
}

