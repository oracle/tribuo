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

package org.tribuo.evaluation;

import org.tribuo.Dataset;
import org.tribuo.MutableDataset;
import org.tribuo.test.MockDataSource;
import org.tribuo.test.MockOutput;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Iterator;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.fail;


public class KFoldSplitterTest {

    private static final Logger logger = Logger.getLogger(KFoldSplitterTest.class.getName());

    @Test
    public void testKFolder() {
        int n = 50;
        int nsplits = 10;
        Dataset<MockOutput> data = getData(n);
        int expectTestSize = n/nsplits;
        int expectTrainSize = n-expectTestSize;
        KFoldSplitter<MockOutput> kf = new KFoldSplitter<>(nsplits, 3);
        Iterator<KFoldSplitter.TrainTestFold<MockOutput>> iter = kf.split(data, true);
        int ct = 0;
        while (iter.hasNext()) {
            KFoldSplitter.TrainTestFold<MockOutput> fold = iter.next();
            assertEquals(expectTrainSize, fold.train.size());
            assertEquals(expectTestSize, fold.test.size());
            ct++;
        }
        assertEquals(nsplits, ct);
    }

    @Test
    public void testKFolderKDoesNotDivideN() {
        int n = 52;
        int nsplits = 10;
        Dataset<MockOutput> data = getData(n);
        int expectTestSize = n/nsplits;
        int expectTrainSize = n-expectTestSize;
        KFoldSplitter<MockOutput> kf = new KFoldSplitter<>(nsplits, 3);
        Iterator<KFoldSplitter.TrainTestFold<MockOutput>> iter = kf.split(data, true);
        int ct = 0;
        while (ct < 2 && iter.hasNext()) {
            KFoldSplitter.TrainTestFold<MockOutput> fold = iter.next();
            assertEquals(expectTrainSize-1, fold.train.size());
            assertEquals(expectTestSize+1, fold.test.size());
            ct++;
        }
        while (iter.hasNext()) {
            KFoldSplitter.TrainTestFold<MockOutput> fold = iter.next();
            assertEquals(expectTrainSize, fold.train.size());
            assertEquals(expectTestSize, fold.test.size());
            ct++;
        }
        assertEquals(nsplits, ct);
    }

    @Test
    public void testKFolderNsplitsGTN() {
        Dataset<MockOutput> data = getData(10);
        KFoldSplitter<MockOutput> kf = new KFoldSplitter<>(11, 3);
        try {
            Iterator<KFoldSplitter.TrainTestFold<MockOutput>> iter = kf.split(data, false);
            fail("should fail for nsplits > ndata");
        } catch (IllegalArgumentException e) {
            //pass
        }
    }

    @Test
    public void testKFolderTwoSplits() {
        Dataset<MockOutput> data = getData(50);
        KFoldSplitter<MockOutput> kf = new KFoldSplitter<>(2,1);
        Iterator<KFoldSplitter.TrainTestFold<MockOutput>> itr = kf.split(data,false);
        while (itr.hasNext()) {
            KFoldSplitter.TrainTestFold<MockOutput> fold = itr.next();
            assertFalse(Arrays.equals(fold.train.getExampleIndices(),fold.test.getExampleIndices()));
        }
    }

    private Dataset<MockOutput> getData(int n) {
        return new MutableDataset<>(new MockDataSource(n));
    }

}
