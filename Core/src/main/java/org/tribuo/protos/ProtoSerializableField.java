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
 * Annotation which denotes that a field should be part of the protobuf serialized representation.
 * <p>
 * Behaviour is undefined when used on a class which doesn't implement {@link ProtoSerializable}.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(FIELD)
public @interface ProtoSerializableField {

    /**
     * The default field name, used to signify it should use the field name rather than a supplied value.
     */
    public static final String DEFAULT_FIELD_NAME = "[DEFAULT_FIELD_NAME]";

    /**
     * The protobuf version when this field was added.
     * @return The version.
     */
    int sinceVersion() default 0;

    /**
     * The name of the field in the protobuf in Java.
     * @return The field name.
     */
    String name() default DEFAULT_FIELD_NAME; 
}
