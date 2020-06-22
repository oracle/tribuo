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

package org.tribuo.common.liblinear;

import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.provenance.ModelProvenance;
import de.bwaldvogel.liblinear.Linear;

import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * A {@link Model} which wraps a LibLinear-java model.
 * <p>
 * It disables the LibLinear debug output as it's very chatty.
 * <p>
 * See:
 * <pre>
 * Fan RE, Chang KW, Hsieh CJ, Wang XR, Lin CJ.
 * "LIBLINEAR: A library for Large Linear Classification"
 * Journal of Machine Learning Research, 2008.
 * </pre>
 * and for the original algorithm:
 * <pre>
 * Cortes C, Vapnik V.
 * "Support-Vector Networks"
 * Machine Learning, 1995.
 * </pre>
 */
public abstract class LibLinearModel<T extends Output<T>> extends Model<T> {
    private static final long serialVersionUID = 3L;

    private static final Logger logger = Logger.getLogger(LibLinearModel.class.getName());

    protected final List<de.bwaldvogel.liblinear.Model> models;

    protected LibLinearModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> labelIDMap, boolean generatesProbabilities, List<de.bwaldvogel.liblinear.Model> models) {
        super(name, description, featureIDMap, labelIDMap, generatesProbabilities);
        this.models = models;
        Linear.disableDebugOutput();
    }

    /**
     * This call is expensive as it copies out the weight matrix from the
     * LibLinear model.
     * <p>
     * Prefer {@link LibLinearModel#getExcuses} to get multiple excuses.
     * <p>
     * @param e The example to excuse.
     * @return An {@link Excuse} for this example.
     */
    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> e) {
        double[][] featureWeights = getFeatureWeights();
        return Optional.of(innerGetExcuse(e, featureWeights));
    }

    @Override
    public Optional<List<Excuse<T>>> getExcuses(Iterable<Example<T>> examples) {
        //This call copies out the weights, so it's better to do it once
        double[][] featureWeights = getFeatureWeights();
        List<Excuse<T>> excuses = new ArrayList<>();

        for (Example<T> e : examples) {
            excuses.add(innerGetExcuse(e, featureWeights));
        }

        return Optional.of(excuses);
    }

    /**
     * Copies the model by writing it out to a String and loading it back in.
     * <p>
     * Unfortunately liblinear-java doesn't have a copy constructor on it's model.
     * @param model THe model to copy.
     * @return A deep copy of the model.
     */
    protected static de.bwaldvogel.liblinear.Model copyModel(de.bwaldvogel.liblinear.Model model) {
        try {
            StringWriter writer = new StringWriter();
            Linear.saveModel(writer,model);
            String modelString = writer.toString();
            StringReader reader = new StringReader(modelString);
            return Linear.loadModel(reader);
        } catch (IOException e) {
            throw new IllegalStateException("IOException found when copying the model in memory via a String.",e);
        }
    }

    /**
     * Extracts the feature weights from the models.
     * The first dimension corresponds to the model index.
     * @return The feature weights.
     */
    protected abstract double[][] getFeatureWeights();

    /**
     * The call to getFeatureWeights in the public methods copies the
     * weights array so this inner method exists to save the copy in getExcuses.
     * <p>
     * If it becomes a problem then we could cache the feature weights in the
     * model.
     * <p>
     * @param e The example.
     * @param featureWeights The per dimension feature weights.
     * @return An excuse for this example.
     */
    protected abstract Excuse<T> innerGetExcuse(Example<T> e, double[][] featureWeights);
}
