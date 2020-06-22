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

package org.tribuo.test;

import org.tribuo.Output;

import java.util.Objects;

/**
 * An Output for use in tests which is very similar to Label.
 */
public class MockOutput implements Output<MockOutput> {
    private static final long serialVersionUID = 1L;

    public final String label;

    public MockOutput(String label) {
        this.label = label;
    }

    @Override
    public MockOutput copy() {
        return new MockOutput(label);
    }

    @Override
    public String getSerializableForm(boolean includeConfidence) {
        return label;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof MockOutput)) return false;

        MockOutput that = (MockOutput) o;

        return label != null ? label.equals(that.label) : that.label == null;
    }

    @Override
    public boolean fullEquals(MockOutput other) {
        return label.equals(other.label);
    }

    @Override
    public String toString() {
        return label;
    }

    @Override
    public int hashCode() {
        return Objects.hash(label);
    }
}

