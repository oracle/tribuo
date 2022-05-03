/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.protos;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import static java.lang.annotation.ElementType.FIELD;

/**
 * Annotation which denotes that the map field this is applied to is
 * serialized as two repeated fields, one for keys and one for values.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(FIELD)
public @interface ProtoSerializableKeysValuesField {
    /**
     * The protobuf version when this field was added.
     * @return The version.
     */
    int sinceVersion() default 0;

    /**
     * The name of the key field in the protobuf in Java.
     * @return The key field name.
     */
    String keysName();

    /**
     * The name of the value field in the protobuf in Java.
     * @return The value field name.
     */
    String valuesName();
}
