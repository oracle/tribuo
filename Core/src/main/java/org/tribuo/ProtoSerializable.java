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

package org.tribuo;

import com.google.protobuf.Message;
import com.oracle.labs.mlrg.olcut.util.Pair;

/**
 * Interface for serializing an implementing object to the specified protobuf.
 * <p>
 * All classes which implement this interface must expose a static method which
 * accepts three arguments (int version, String className, com.google.protobuf.Any message)
 * and returns an instance of this class.
 * We can't require this with the type system yet, so it must be checked by tests.
 * <p>
 * The deserialization factory is accessed reflectively, and so if it is not public
 * the module must be opened to the {@code org.tribuo.core} module.
 * <p>
 * If the type being deserialized changes its name then a redirect can be added via
 * {@link org.tribuo.util.ProtoUtil#registerRedirect(Pair, String)}.
 * @param <T> The protobuf type.
 */
public interface ProtoSerializable<T extends Message> {

    /**
     * Serializes this object to a protobuf.
     * @return The protobuf
     */
    public T serialize();

}
