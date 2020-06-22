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

package org.tribuo.util.infotheory.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A row of values from a {@link RowList}.
 * <p>
 * Rows are defined with a hashcode and equals based on their contained values,
 * and are immutable. They interact with the information theory calculations
 * via the equals method.
 * @param <T> The type of values.
 */
public final class Row<T> {
    private final List<T> innerRow;

    Row(List<T> innerRow) {
        this.innerRow = Collections.unmodifiableList(new ArrayList<>(innerRow));
    }   

    @Override
    public boolean equals(Object other) {
        if (other instanceof Row) {
            Row<?> otherRow = (Row<?>) other;
            if (otherRow.innerRow.size() == innerRow.size()) {
                boolean check = true;
                for (int i = 0; i < innerRow.size(); i++) {
                    check = check && innerRow.get(i).equals(otherRow.innerRow.get(i));
                }
                return check;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        int hash = 42;
        for (T t : innerRow) {
            hash ^= t.hashCode();
        }
        return hash;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("Row = (");
        for (T element : innerRow) {
            builder.append(element.toString());
            builder.append(',');
        }
        builder.deleteCharAt(builder.length()-1);
        builder.append(')');
        return builder.toString();
    }
}
