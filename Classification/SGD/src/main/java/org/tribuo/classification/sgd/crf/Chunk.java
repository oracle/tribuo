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

package org.tribuo.classification.sgd.crf;

/**
 * Chunk class used for chunk level confidence prediction in the {@link CRFModel}.
 */
public class Chunk {
    public final int begin;
    public final int[] labels;

    public Chunk(int begin, int[] labels) {
        this.begin = begin;
        this.labels = labels;
    }

    public void unpack(int[] constraints) {
        System.arraycopy(labels, 0, constraints, begin, labels.length);
    }
}
