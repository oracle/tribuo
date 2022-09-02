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

/**
 * Classes which control the serialization of Tribuo objects to and from protocol buffers.
 * <p>
 * There is an automatic serialization mechanism which requires the fields to be
 * annotated with {@link org.tribuo.protos.ProtoSerializableField} and similar. The
 * enclosing class must implement {@link org.tribuo.protos.ProtoSerializable} and
 * be annotated with {@link org.tribuo.protos.ProtoSerializableClass}. This mechanism
 * can serialize:
 * <ul>
 *     <li>primitive types</li>
 *     <li>Strings</li>
 *     <li>classes which implement {@link org.tribuo.protos.ProtoSerializable}</li>
 *     <li>lists, sets and maps of supported types</li>
 * </ul>
 * <p>
 * Maps may be serialized in multiple ways, as protobuf does not support the full set of
 * map type parameters that Java does. The variants are:
 * <ul>
 *     <li>{@link org.tribuo.protos.ProtoSerializableMapField} annotation directly serializes the map as a protobuf map</li>
 *     <li>{@link org.tribuo.protos.ProtoSerializableKeysValuesField} serializes the map as two repeated fields for the keys and values</li>
 *     <li>{@link org.tribuo.protos.ProtoSerializableMapValuesField} serializes just the values from the map</li>
 * </ul>
 */
package org.tribuo.protos;