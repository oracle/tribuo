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

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;

/**
 * An implementation of a List which wraps a set of lists.
 * <p>
 * Each access returns a {@link Row} drawn by taking an element from each list.
 * <p>
 * The rows only expose equals and hashcode, as the information theoretic calculations
 * only care about equality.
 * @param <T> The type stored in the lists.
 */
public final class RowList<T> implements List<Row<T>> {
    private final Set<List<T>> set;
    private final int size;

    /**
     * Constructs a RowList from a set of lists.
     * @param set The feature lists.
     */
    public RowList(Set<List<T>> set) {
        this.set = Collections.unmodifiableSet(new LinkedHashSet<>(set));
        size = set.iterator().next().size();
        for (Collection<T> element : this.set) {
            if (size != element.size()) {
                throw new IllegalArgumentException("Not all the collections in the set are the same length");
            }
        }
    }
    
    @Override
    public int size() {
        return size;
    }

    @Override
    public boolean isEmpty() {
        return size == 0;
    }

    @Override
    public boolean contains(Object o) {
        if (o instanceof Row) {
            Row<?> otherRow = (Row<?>) o;
            boolean found = false;
            for (Row<T> row : this) {
                if (otherRow.equals(row)) {
                    found = true;
                    break;
                }
            }
            return found;
        } else {
            return false;
        }
    }

    @Override
    public Iterator<Row<T>> iterator() {
        return new RowListIterator<>(set);
    }

    @Override
    public Object[] toArray() {
        Object[] output = new Object[size];
        int counter = 0;
        for (Row<T> row : this) {
            output[counter] = row;
            counter++;
        }
        return output;
    }

    @Override
    @SuppressWarnings("unchecked")
    public <U> U[] toArray(U[] a) {
        U[] output = a;
        if (output.length < size) {
            output = (U[]) Array.newInstance(a[0].getClass(), size);
        }
        int counter = 0;
        for (Row<T> row : this) {
            output[counter] = (U) row;
            counter++;
        }
        if (output.length > size) {
            //fill with nulls if bigger.
            for (; counter < output.length; counter++) {
                output[counter] = null;
            }
        }
        return output;
    }

    @Override
    public Row<T> get(int index) {
        ArrayList<T> list = new ArrayList<>(set.size());
        int counter = 0;
        for (List<T> element : set) {
            list.add(counter, element.get(index));
            counter++;
        }
        return new Row<>(list);
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        boolean found = true;
        Iterator<?> itr = c.iterator();
        while (itr.hasNext() && found) {
            found = this.contains(itr.next());
        }
        return found;
    }

    @Override
    public int indexOf(Object o) {
        if (o instanceof Row) {
            Row<?> otherRow = (Row<?>) o;
            int counter = 0;
            int found = -1;
            Iterator<Row<T>> itr = this.iterator();
            while (itr.hasNext() && found == -1) {
                if (itr.next().equals(otherRow)) {
                    found = counter;
                }
                counter++;
            }
            return found;
        } else {
            return -1;
        }
    }

    @Override
    public int lastIndexOf(Object o) {
        if (o instanceof Row) {
            Row<?> otherRow = (Row<?>) o;
            int counter = 0;
            int found = -1;
            for (Row<T> tRow : this) {
                if (tRow.equals(otherRow)) {
                    found = counter;
                }
                counter++;
            }
            return found;
        } else {
            return -1;
        }
    }

    @Override
    public ListIterator<Row<T>> listIterator() {
        return new RowListIterator<>(set);
    }

    @Override
    public ListIterator<Row<T>> listIterator(int index) {
        return new RowListIterator<>(set,index);
    }

    /**
     * Unsupported. Throws UnsupportedOperationException.
     * @param fromIndex n/a
     * @param toIndex n/a
     * @return n/a
     */
    @Override
    public List<Row<T>> subList(int fromIndex, int toIndex) {
        throw new UnsupportedOperationException("Views are not supported on a RowList.");
    }

    //*************************************************************************
    // The remaining operations are unsupported as this list is immutable.
    //*************************************************************************
    /**
     * Unsupported. Throws UnsupportedOperationException.
     * @param e n/a
     * @return n/a
     */
    @Override
    public boolean add(Row<T> e) {
        throw new UnsupportedOperationException("This list is immutable.");
    }

    /**
     * Unsupported. Throws UnsupportedOperationException.
     * @param o n/a
     * @return n/a
     */
    @Override
    public boolean remove(Object o) {
        throw new UnsupportedOperationException("This list is immutable.");
    }

    /**
     * Unsupported. Throws UnsupportedOperationException.
     * @param c n/a
     * @return n/a
     */
    @Override
    public boolean addAll(Collection<? extends Row<T>> c) {
        throw new UnsupportedOperationException("This list is immutable.");
    }

    /**
     * Unsupported. Throws UnsupportedOperationException.
     * @param index n/a
     * @param c n/a
     * @return n/a
     */
    @Override
    public boolean addAll(int index, Collection<? extends Row<T>> c) {
        throw new UnsupportedOperationException("This list is immutable.");
    }

    /**
     * Unsupported. Throws UnsupportedOperationException.
     * @param c n/a
     * @return n/a
     */
    @Override
    public boolean removeAll(Collection<?> c) {
        throw new UnsupportedOperationException("This list is immutable.");
    }

    /**
     * Unsupported. Throws UnsupportedOperationException.
     * @param c n/a
     * @return n/a
     */
    @Override
    public boolean retainAll(Collection<?> c) {
        throw new UnsupportedOperationException("This list is immutable.");
    }

    /**
     * Unsupported. Throws UnsupportedOperationException.
     */
    @Override
    public void clear() {
        throw new UnsupportedOperationException("This list is immutable.");
    }

    /**
     * Unsupported. Throws UnsupportedOperationException.
     * @param index n/a
     * @param element n/a
     * @return n/a
     */
    @Override
    public Row<T> set(int index, Row<T> element) {
        throw new UnsupportedOperationException("This list is immutable.");
    }

    /**
     * Unsupported. Throws UnsupportedOperationException.
     * @param index n/a
     * @param element n/a
     */
    @Override
    public void add(int index, Row<T> element) {
        throw new UnsupportedOperationException("This list is immutable.");
    }

    /**
     * Unsupported. Throws UnsupportedOperationException.
     * @param index n/a
     * @return n/a
     */
    @Override
    public Row<T> remove(int index) {
        throw new UnsupportedOperationException("This list is immutable.");
    }

    /**
     * The iterator over the rows.
     * @param <T> The type of the row.
     */
    private static class RowListIterator<T> implements ListIterator<Row<T>> {
        private int curIndex;
        private final int size;
        private final Set<List<T>> set;

        public RowListIterator(Set<List<T>> set) {
            this(set,0);
        }

        public RowListIterator(Set<List<T>> set, int curIndex) {
            this.curIndex = curIndex;
            this.set = set;
            this.size = set.iterator().next().size();
        }

        @Override
        public boolean hasNext() {
            return curIndex < size;
        }

        @Override
        public Row<T> next() {
            ArrayList<T> list = new ArrayList<>(set.size());
            int counter = 0;
            for (List<T> element : set) {
                list.add(counter, element.get(curIndex));
                counter++;
            }
            curIndex++;
            return new Row<>(list);
        }

        @Override
        public boolean hasPrevious() {
            return curIndex > 0;
        }

        @Override
        public Row<T> previous() {
            ArrayList<T> list = new ArrayList<>(set.size());
            curIndex--;
            int counter = 0;
            for (List<T> element : set) {
                list.add(counter, element.get(curIndex));
                counter++;
            }
            return new Row<>(list);
        }

        @Override
        public int nextIndex() {
            return curIndex;
        }

        @Override
        public int previousIndex() {
            return curIndex - 1;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("The list backing this iterator is immutable.");
        }

        @Override
        public void set(Row<T> e) {
            throw new UnsupportedOperationException("The list backing this iterator is immutable.");
        }

        @Override
        public void add(Row<T> e) {
            throw new UnsupportedOperationException("The list backing this iterator is immutable.");
        }
    }
}

