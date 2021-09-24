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

package org.tribuo.provenance;

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;

/**
 * Data source provenance.
 */
public interface DataSourceProvenance extends DataProvenance, ObjectProvenance {

    /**
     * The name of the provenance field for the resource hash.
     */
    public static final String RESOURCE_HASH = "resource-hash";

    /**
     * The name of the provenance field for the file timestamp.
     */
    public static final String FILE_MODIFIED_TIME = "file-modified-time";

    /**
     * The name of the provenance field for the datasource timestamp.
     */
    public static final String DATASOURCE_CREATION_TIME = "datasource-creation-time";

    /**
     * The name of the provenance field for the output factory.
     */
    public static final String OUTPUT_FACTORY = "outputFactory";

}
