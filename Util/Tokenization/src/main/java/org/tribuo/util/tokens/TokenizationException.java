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

package org.tribuo.util.tokens;

/**
 * Wraps exceptions thrown by tokenizers.
 */
public class TokenizationException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    /**
     * Creates a TokenizationException with the specified message.
     * @param message The exception message.
     */
    public TokenizationException(String message) {
        super(message);
    }

    /**
     * Creates a TokenizationException wrapping the supplied throwable with the specified message.
     * @param message The exception message.
     * @param throwable The throwable to wrap.
     */
    public TokenizationException(String message, Throwable throwable) {
        super(message,throwable);
    }

    /**
     * Creates a TokenizationException wrapping the supplied throwable.
     * @param throwable The throwable to wrap.
     */
    public TokenizationException(Throwable throwable) {
        super(throwable);
    }
}
