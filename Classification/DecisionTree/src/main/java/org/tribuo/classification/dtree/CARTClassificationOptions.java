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

package org.tribuo.classification.dtree;

import com.oracle.labs.mlrg.olcut.config.Option;
import org.tribuo.Trainer;
import org.tribuo.classification.ClassificationOptions;
import org.tribuo.classification.dtree.impurity.Entropy;
import org.tribuo.classification.dtree.impurity.GiniIndex;
import org.tribuo.classification.dtree.impurity.LabelImpurity;

/**
 * Options for building a classification tree trainer.
 */
public class CARTClassificationOptions implements ClassificationOptions<CARTClassificationTrainer> {

    @Override
    public String getOptionsDescription() {
        return "Options for decision/classification trees.";
    }

    public enum TreeType {CART}

    public enum ImpurityType {GINI, ENTROPY}

    @Option(longName = "cart-max-depth", usage = "Maximum depth in the decision tree.")
    public int cartMaxDepth = 6;
    @Option(longName = "cart-min-child-weight", usage = "Minimum child weight.")
    public float cartMinChildWeight = 5.0f;
    @Option(longName = "cart-split-fraction", usage = "Fraction of features in split.")
    public float cartSplitFraction = 0.0f;
    @Option(longName = "cart-impurity", usage = "Impurity measure to use. Defaults to GINI.")
    public ImpurityType cartImpurity = ImpurityType.GINI;
    @Option(longName = "cart-print-tree", usage = "Prints the decision tree.")
    public boolean cartPrintTree;
    @Option(longName = "cart-tree-algorithm", usage = "Tree algorithm to use (options are CART).")
    public TreeType cartTreeAlgorithm = TreeType.CART;
    @Option(longName = "cart-seed", usage = "RNG seed.")
    public long cartSeed = Trainer.DEFAULT_SEED;

    @Override
    public CARTClassificationTrainer getTrainer() {
        LabelImpurity impurity;
        switch (cartImpurity) {
            case GINI:
                impurity = new GiniIndex();
                break;
            case ENTROPY:
                impurity = new Entropy();
                break;
            default:
                throw new IllegalArgumentException("unknown impurity type " + cartImpurity);
        }

        CARTClassificationTrainer trainer;
        switch (cartTreeAlgorithm) {
            case CART:
                if (cartSplitFraction <= 0) {
                    trainer = new CARTClassificationTrainer(cartMaxDepth, cartMinChildWeight, 1, impurity, cartSeed);
                } else {
                    trainer = new CARTClassificationTrainer(cartMaxDepth, cartMinChildWeight, cartSplitFraction, impurity, cartSeed);
                }
                break;
            default:
                throw new IllegalArgumentException("Unknown tree type " + cartTreeAlgorithm);
        }

        return trainer;
    }

}
