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

package org.tribuo.transform.transformations;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.IDFTransformerProto;
import org.tribuo.protos.core.TransformerProto;
import org.tribuo.transform.Transformer;
import org.tribuo.transform.transformations.IDFTransformation.IDFTransformer;

public class IDFTransformTest {

    @Test
    void testSerialize() throws Exception {
        IDFTransformer idft = new IDFTransformer(13, 17);
        TransformerProto tp = idft.serialize();
        assertEquals(0, tp.getVersion());
        assertEquals("org.tribuo.transform.transformations.IDFTransformation$IDFTransformer", tp.getClassName());
        IDFTransformerProto proto = tp.getSerializedData().unpack(IDFTransformerProto.class);
        assertEquals(13, proto.getDf());
        assertEquals(17, proto.getN());

        Transformer tD = ProtoUtil.deserialize(tp);
        assertEquals(idft, tD);
    }

}
