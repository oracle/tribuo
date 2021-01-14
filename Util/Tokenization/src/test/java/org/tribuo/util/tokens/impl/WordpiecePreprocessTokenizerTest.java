package org.tribuo.util.tokens.impl;

import org.junit.jupiter.api.Test;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.TokenizerTestBase;
import org.tribuo.util.tokens.impl.wordpiece.WordpiecePreprocessTokenizer;

public class WordpiecePreprocessTokenizerTest extends TokenizerTestBase {

    @Test
    public void testBasic() {
        Tokenizer tokenizer = new WordpiecePreprocessTokenizer();
        test(tokenizer, "a b  c\nd\t\te \t\r\nf", "a", "b", "c", "d", "e", "f");
        test(tokenizer, "");
        test(tokenizer, " \t\r\n");
        test(tokenizer, " \t\r\na", "a");
        test(tokenizer, "a \t\r\n", "a");
        test(tokenizer, "abcd", "abcd");
        test(tokenizer, "abcd efgh", "abcd", "efgh");
        test(tokenizer, "hello world!", "hello", "world", "!");
        test(tokenizer, "hello-world!!!goodbye  earth. p", "hello", "-", "world", "!", "!", "!", "goodbye", "earth", ".", "p");
        test(tokenizer, "8 km", "8", "km");
        test(tokenizer, "238.8 km", "238", ".", "8", "km");
        test(tokenizer, "金泰均", "金", "泰", "均");
        test(tokenizer, "का प्रयोग", "क", "ा", "प", "र", "य", "ो", "ग");
    }

}
