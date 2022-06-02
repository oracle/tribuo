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

import com.google.protobuf.Message;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import static java.lang.annotation.ElementType.TYPE;

/**
 * Mark a class as being {@link ProtoSerializable} and specify
 * the class type used to serialize the "serialized_data".
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(TYPE)
public @interface ProtoSerializableClass {
    /**
     * Specifies the type of the serialized data payload.
     * @return The proto class of the serialized data.
     */
    Class<? extends Message> serializedDataClass() default Message.class;

    /**
     * The current version of this proto serialized class.
     * @return The version number.
     */
    int version();
}
