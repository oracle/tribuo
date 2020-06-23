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

package org.tribuo.datasource;

import org.tribuo.DataSource;
import org.tribuo.Example;
import org.tribuo.FeatureMap;
import org.tribuo.MutableDataset;
import org.tribuo.Output;
import org.tribuo.OutputInfo;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 *
 */
public class LibSVMDataSourceTest {

    @Test
    public void testLibSVMLoading() throws IOException {
        MockOutputFactory factory = new MockOutputFactory();
        URL dataFile = LibSVMDataSourceTest.class.getResource("/org/tribuo/datasource/test-1.libsvm");
        LibSVMDataSource<MockOutput> source = new LibSVMDataSource<>(dataFile,factory);

        MutableDataset<MockOutput> dataset = new MutableDataset<>(source);

        FeatureMap fmap = dataset.getFeatureMap();
        assertEquals(7,fmap.size());

        OutputInfo<MockOutput> info = dataset.getOutputInfo();
        assertEquals(2,info.size());
    }

    @Test
    public void testLibSVMSaving() throws IOException {
        MockOutputFactory factory = new MockOutputFactory();

        URL dataFile = LibSVMDataSourceTest.class.getResource("/org/tribuo/datasource/test-1.libsvm");
        LibSVMDataSource<MockOutput> source = new LibSVMDataSource<>(dataFile,factory);

        File temp = File.createTempFile("tribuo-lib-svm-test","libsvm");
        temp.deleteOnExit();

        MutableDataset<MockOutput> dataset = new MutableDataset<>(source);

        PrintStream stream = new PrintStream(temp, StandardCharsets.UTF_8.name());
        LibSVMDataSource.writeLibSVMFormat(dataset,stream,false, (MockOutput a) -> Integer.parseInt(a.label));
        stream.close();

        LibSVMDataSource<MockOutput> loadedSource = new LibSVMDataSource<>(temp.toPath(),factory);

        assertTrue(compareDataSources(source,loadedSource), "Saved data source was not the same as the loaded one.");
    }

    public static <T extends Output<T>> boolean compareDataSources(DataSource<T> first, DataSource<T> second) {
        boolean same = true;

        Iterator<Example<T>> firstItr = first.iterator();
        Iterator<Example<T>> secondItr = second.iterator();

        while (firstItr.hasNext() && secondItr.hasNext() && same) {
            Example<T> firstExample = firstItr.next();
            Example<T> secondExample = secondItr.next();

            same &= firstExample.equals(secondExample);
        }

        if (firstItr.hasNext() || secondItr.hasNext()) {
            // an iterator is not exhausted, either one of the iterators is short
            // or an example was not the same
            return false;
        } else {
            return same;
        }

    }

}
