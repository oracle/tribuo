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

import java.io.Serializable;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * A triple of things. The inner pairs are cached, as is the hashcode.
 * <p>
 * The cache is calculated on construction, and the objects inside the triple are thus expected to be immutable.
 * If they aren't then the behaviour is undefined (and you shouldn't use this class).
 * @param <T1> The type of the first object.
 * @param <T2> The type of the second object.
 * @param <T3> The type of the third object.
 */
public class CachedTriple<T1, T2, T3> implements Serializable {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(CachedTriple.class.getName());

    private final CachedPair<T1,T2> ab;
    private final CachedPair<T1,T3> ac;
    private final CachedPair<T2,T3> bc;

    /**
     * The first element.
     */
    protected final T1 a;
    /**
     * The second element.
     */
    protected final T2 b;
    /**
     * The third element.
     */
    protected final T3 c;

    private final int cachedHash;

    /**
     * Constructs a CachedTriple.
     * @param a The first element.
     * @param b The second element.
     * @param c The third element.
     */
    public CachedTriple(T1 a, T2 b, T3 c) {
        this.a = a;
        this.b = b;
        this.c = c;
        this.ab = new CachedPair<>(a,b);
        this.ac = new CachedPair<>(a,c);
        this.bc = new CachedPair<>(b,c);
        this.cachedHash = calculateHashCode();
    }

    /**
     * Gets the first element.
     * @return The first element.
     */
    public T1 getA() {
        return a;
    }

    /**
     * Gets the second element.
     * @return The second element.
     */
    public T2 getB() {
        return b;
    }

    /**
     * Gets the third element.
     * @return The third element.
     */
    public T3 getC() {
        return c;
    }

    /**
     * Gets the pair of the first and second elements.
     * @return A pair of the first and second elements.
     */
    public CachedPair<T1,T2> getAB() {
        return ab;
    }

    /**
     * Gets the pair of the first and third elements.
     * @return A pair of the first and third elements.
     */
    public CachedPair<T1,T3> getAC() {
        return ac;
    }

    /**
     * Gets the pair of the second and third elements.
     * @return A pair of the second and third elements.
     */
    public CachedPair<T2,T3> getBC() {
        return bc;
    }

    /**
     * Used to mix the integers in hashcode.
     * Returns the 32 high bits of Stafford variant 4 mix64 function as int.
     */
    private static int mix32(long z) {
        z *= 0x62a9d9ed799705f5L;
        return (int)(((z ^ (z >>> 28)) * 0xcb24d0a5c88c35b3L) >>> 32);
    }

    @Override
    public int hashCode() {
        return cachedHash;
    }
    
    /**
     * Overridden hashcode. Checks to see if the types are ints or longs, and
     * runs them through the mixing function if
     * they are. Then XORs the two hashcodes together.
     * @return A 32-bit integer.
     */
    public int calculateHashCode() {
        int aCode, bCode, cCode;
        
        if (a instanceof Integer) {
            aCode = mix32((Integer) a);
        } else if (a instanceof Long) {
            aCode = mix32((Long) a);
        } else {
            aCode = a.hashCode();
        }

        if (b instanceof Integer) {
            bCode = mix32((Integer) b);
        } else if (b instanceof Long) {
            bCode = mix32((Long) b);
        } else {
            bCode = b.hashCode();
        }

        if (c instanceof Integer) {
            cCode = mix32((Integer) c);
        } else if (c instanceof Long) {
            cCode = mix32((Long) c);
        } else {
            cCode = c.hashCode();
        }
        
        return (aCode ^ bCode) ^ cCode;
    }

    @Override
    public boolean equals(Object obj) {
        if(obj == null) {
            return false;
        }
        if(!(obj instanceof CachedTriple)) {
            return false;
        }
        final CachedTriple<?,?,?> other = (CachedTriple<?,?,?>) obj;
        if(!Objects.equals(this.a, other.a)) {
            return false;
        }
        if(!Objects.equals(this.b, other.b)) {
            return false;
        }
        return Objects.equals(this.c, other.c);
    }

    @Override
    public String toString() {
        return "Triple{" + "a=" + a + ", b=" + b + ", c=" + c + '}';
    }
}
