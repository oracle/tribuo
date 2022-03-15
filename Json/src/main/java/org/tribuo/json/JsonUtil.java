/*
 * Copyright (c) 2020, 2022 Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Utilities for interacting with JSON objects or text representations.
 */
public final class JsonUtil {

    private static final Logger logger = Logger.getLogger(JsonUtil.class.getName());

    /**
     * Final class with private constructor
     */
    private JsonUtil() {}

    /**
     * Converts a Json node into a Map from String to String for use in
     * downstream processing by {@link org.tribuo.data.columnar.RowProcessor}.
     * <p>
     * This method ignores any fields which are not primitives (i.e., it ignores
     * fields which are arrays and objects) as those are not supported
     * by the columnar processing infrastructure.
     * <p>
     * If the node is null it returns Collections#emptyMap.
     * @param node The json object to convert.
     * @return The map representing this json node.
     */
    public static Map<String,String> convertToMap(ObjectNode node) {
        if (node != null) {
            Map<String,String> entry = new HashMap<>();
            for (Iterator<Map.Entry<String, JsonNode>> itr = node.fields(); itr.hasNext(); ) {
                Map.Entry<String, JsonNode> e = itr.next();
                if (e.getValue() != null) {
                    if (e.getValue().isValueNode()) {
                        entry.put(e.getKey(), e.getValue().asText());
                    } else {
                        logger.warning("Ignoring key '" + e.getKey() + "' as it's value '" + e.getValue().asText() + "' is an object or array");
                    }
                }
            }
            return entry;
        } else {
            return Collections.emptyMap();
        }
    }
}
