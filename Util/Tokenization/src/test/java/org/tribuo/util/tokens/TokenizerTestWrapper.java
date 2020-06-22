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

import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.util.IOUtil;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class TokenizerTestWrapper implements Serializable {
    private static final long serialVersionUID = 1L;
    public String name;
    public Tokenizer tokenizer;

    public TokenizerTestWrapper(String name, Tokenizer tokenizer) {
        super();
        this.name = name;
        this.tokenizer = tokenizer;
    }

    public static Tokenizer serializeAndDeserialize(File f, Tokenizer tokenizer) throws IOException, ClassNotFoundException {
        TokenizerTestWrapper mtc = new TokenizerTestWrapper("test-my-tokenizer", tokenizer);
        IOUtil.serialize(mtc, f.getPath());
        mtc = IOUtil.deserialize(f.getPath(), TokenizerTestWrapper.class).get();
        tokenizer = mtc.tokenizer;
        return tokenizer;
    }

    private void readObject(ObjectInputStream inputStream) throws ClassNotFoundException, IOException {
        this.name = (String) inputStream.readObject();
        this.tokenizer = (Tokenizer) ProvenanceUtil.readObject(inputStream);
    }

    private void writeObject(ObjectOutputStream outputStream) throws IOException {
        outputStream.writeObject(this.name);
        ProvenanceUtil.writeObject(this.tokenizer, outputStream);
    }
}
