package org.tribuo.util.tokens.impl;

import org.junit.jupiter.api.Test;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.TokenizerTestBase;

public class WhitespaceTokenizerTest extends TokenizerTestBase {

    @Test
    public void testBasic() {
        Tokenizer tokenizer = new WhitespaceTokenizer();
        test(tokenizer, "a b  c\nd\t\te \t\r\nf", "a", "b", "c", "d", "e", "f");
        test(tokenizer, "");
        test(tokenizer, " \t\r\n");
        test(tokenizer, " \t\r\na", "a");
        test(tokenizer, "a \t\r\n", "a");
        test(tokenizer, "abcd", "abcd");
        test(tokenizer, "abcd efgh", "abcd", "efgh");
    }

}
