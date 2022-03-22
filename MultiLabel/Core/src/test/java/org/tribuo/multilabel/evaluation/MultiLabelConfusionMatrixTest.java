/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.multilabel.evaluation;

import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.multilabel.MultiLabel;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.tribuo.multilabel.Utils.getUnknown;
import static org.tribuo.multilabel.Utils.label;
import static org.tribuo.multilabel.Utils.mkDomain;
import static org.tribuo.multilabel.Utils.mkPrediction;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class MultiLabelConfusionMatrixTest {

    @Test
    public void testTabulateUnexpectedMultiLabel() {
        MultiLabel a = label("a");
        MultiLabel bc = label("b","c");
        MultiLabel abd = label("a","b","d");
        ImmutableOutputInfo<MultiLabel> domain = mkDomain(a,bc,abd);

        List<Prediction<MultiLabel>> predictions = Arrays.asList(
                mkPrediction(label("a"), label("a")),
                mkPrediction(label("c"), label("b")),
                mkPrediction(label("b"), label("b")),
                //Note "e" is not in the domain.
                mkPrediction(label("e"), label("a", "c"))
        );

        try {
            MultiLabelConfusionMatrix.ConfusionMatrixTuple t = MultiLabelConfusionMatrix.tabulate(domain, predictions);
            fail("Should have thrown IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Unknown label 'e' found"));
        }
    }

    @Test
    public void testTabulateUnknownMultiLabel() {
        List<Prediction<MultiLabel>> predictions = Arrays.asList(
                mkPrediction(getUnknown(), label("a")),
                mkPrediction(label("c"), label("b")),
                mkPrediction(label("b"), label("b")),
                mkPrediction(label("b"), label("c"))
        );
        ImmutableOutputInfo<MultiLabel> domain = mkDomain(label("a"),label("b"),label("c"));

        try {
            MultiLabelConfusionMatrix.ConfusionMatrixTuple t = MultiLabelConfusionMatrix.tabulate(domain, predictions);
            fail("Should have thrown IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("sentinel Unknown MultiLabel"));
        }

        predictions = Arrays.asList(
                mkPrediction(label("a"), label("a")),
                mkPrediction(label("c"), getUnknown()),
                mkPrediction(label("b"), label("b")),
                mkPrediction(label("b"), label("c"))
        );

        try {
            MultiLabelConfusionMatrix.ConfusionMatrixTuple t = MultiLabelConfusionMatrix.tabulate(domain, predictions);
            fail("Should have thrown IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("sentinel Unknown MultiLabel"));
        }
    }

    @Test
    public void testTabulateSingleLabel() {
        MultiLabel a = label("a");
        MultiLabel b = label("b");
        MultiLabel c = label("c");
        List<Prediction<MultiLabel>> predictions = Arrays.asList(
            mkPrediction(a, a),
            mkPrediction(c, b),
            mkPrediction(b, b),
            mkPrediction(b, c)
        );
        ImmutableOutputInfo<MultiLabel> domain = mkDomain(predictions);

        DenseMatrix[] mcm = MultiLabelConfusionMatrix
                .tabulate(domain, predictions)
                .getMCM();

        int aIndex = domain.getID(a);
        int bIndex = domain.getID(b);
        int cIndex = domain.getID(c);

        assertEquals(domain.size(), mcm.length);

        assertEquals(3d, mcm[aIndex].get(0, 0));
        assertEquals(1d, mcm[aIndex].get(1, 1));

        assertEquals(1d, mcm[bIndex].get(0, 0));
        assertEquals(1d, mcm[bIndex].get(0, 1));
        assertEquals(1d, mcm[bIndex].get(1, 0));
        assertEquals(1d, mcm[bIndex].get(1, 1));

        assertEquals(2d, mcm[cIndex].get(0, 0));
        assertEquals(1d, mcm[cIndex].get(0, 1));
        assertEquals(1d, mcm[cIndex].get(1, 0));
    }

    /**
     * Technically, these are single-label outputs; check we get the same results as we do for LabelConfusionMatrix.
     */
    @Test
    public void testSingleLabel() {
        List<Prediction<MultiLabel>> predictions = Arrays.asList(
                mkPrediction(label("a"), label("a")),
                mkPrediction(label("c"), label("b")),
                mkPrediction(label("b"), label("b")),
                mkPrediction(label("b"), label("c"))
        );

        ImmutableOutputInfo<MultiLabel> domain = mkDomain(predictions);
        assertEquals(3, domain.size());

        MultiLabelConfusionMatrix cm = new MultiLabelConfusionMatrix(domain, predictions);

        MultiLabel a = label("a");
        MultiLabel b = label("b");
        MultiLabel c = label("c");

        assertEquals(1, cm.tp(a));
        assertEquals(0, cm.fp(a));
        assertEquals(3, cm.tn(a));
        assertEquals(0, cm.fn(a));
        assertEquals(1, cm.support(a));

        assertEquals(1, cm.tp(b));
        assertEquals(1, cm.fp(b));
        assertEquals(1, cm.tn(b));
        assertEquals(1, cm.fn(b));
        assertEquals(2, cm.support(b));

        assertEquals(0, cm.tp(c));
        assertEquals(1, cm.fp(c));
        assertEquals(2, cm.tn(c));
        assertEquals(1, cm.fn(c));
        assertEquals(1, cm.support(c));

        assertEquals(4, cm.support());
    }

    @Test
    public void testTabulateMultiLabel() {
        MultiLabel a = label("a");
        MultiLabel b = label("b");
        MultiLabel c = label("c");
        List<Prediction<MultiLabel>> predictions = Arrays.asList(
                mkPrediction(label("a"), label("a", "b")),
                mkPrediction(label("c", "b"), label("b")),
                mkPrediction(label("b"), label("b")),
                mkPrediction(label("b"), label("c"))
        );

        ImmutableOutputInfo<MultiLabel> domain = mkDomain(predictions);
        assertEquals(3, domain.size());

        DenseMatrix[] mcm = MultiLabelConfusionMatrix
                .tabulate(domain, predictions)
                .getMCM();

        int aIndex = domain.getID(a);
        int bIndex = domain.getID(b);
        int cIndex = domain.getID(c);

        assertEquals(domain.size(), mcm.length);
        assertEquals(3d, mcm[aIndex].get(0, 0));
        assertEquals(1d, mcm[aIndex].get(1, 1));

        assertEquals(1d, mcm[bIndex].get(0, 1));
        assertEquals(1d, mcm[bIndex].get(1, 0));
        assertEquals(2d, mcm[bIndex].get(1, 1));

        assertEquals(2d, mcm[cIndex].get(0, 0));
        assertEquals(1d, mcm[cIndex].get(0, 1));
        assertEquals(1d, mcm[cIndex].get(1, 0));
    }

    @Test
    public void testMultiLabel() {
        List<Prediction<MultiLabel>> predictions = Arrays.asList(
                mkPrediction(label("a"), label("a", "b")),
                mkPrediction(label("c", "b"), label("b")),
                mkPrediction(label("b"), label("b")),
                mkPrediction(label("b"), label("c"))
        );

        ImmutableOutputInfo<MultiLabel> domain = mkDomain(predictions);
        MultiLabelConfusionMatrix cm = new MultiLabelConfusionMatrix(domain, predictions);

        MultiLabel a = label("a");
        MultiLabel b = label("b");
        MultiLabel c = label("c");

        assertEquals(1, cm.confusion(a, a));
        assertEquals(0, cm.confusion(a, b));
        assertEquals(1, cm.confusion(b, a));
        assertEquals(1, cm.confusion(b, c));
        assertEquals(2, cm.confusion(b, b));
        assertEquals(1, cm.confusion(c, b));

        assertEquals(1, cm.tp(a));
        assertEquals(0, cm.fp(a));
        assertEquals(3, cm.tn(a));
        assertEquals(0, cm.fn(a));
        assertEquals(1, cm.support(a));

        assertEquals(2, cm.tp(b));
        assertEquals(1, cm.fp(b));
        assertEquals(0, cm.tn(b));
        assertEquals(1, cm.fn(b));
        assertEquals(3, cm.support(b));

        assertEquals(0, cm.tp(c));
        assertEquals(1, cm.fp(c));
        assertEquals(2, cm.tn(c));
        assertEquals(1, cm.fn(c));
        assertEquals(1, cm.support(c));

        assertEquals(5, cm.support());
    }


}