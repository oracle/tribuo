/*
 * Copyright (c) 2021 Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop.tensorflow.example;

import org.tensorflow.proto.framework.GraphDef;

/**
 * A tuple containing a graph def protobuf along with the relevant operation names.
 * <p>
 * Will be a record one day.
 */
public final class GraphDefTuple {
    public final GraphDef graphDef;
    public final String inputName;
    public final String outputName;
    public final String initName;

    /**
     * Creates a graphDef record.
     * @param graphDef The TF Graph.
     * @param inputName The name of the input placeholder.
     * @param outputName The name of the output operation.
     * @param initName The name of the init operation.
     */
    public GraphDefTuple(GraphDef graphDef, String inputName, String outputName, String initName) {
        this.graphDef = graphDef;
        this.initName = initName;
        this.outputName = outputName;
        this.inputName = inputName;
    }
}
