package org.tribuo.util.tokens.impl;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.params.provider.Arguments.arguments;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.TokenizerTestBase;
import org.tribuo.util.tokens.impl.wordpiece.Wordpiece;
import org.tribuo.util.tokens.impl.wordpiece.WordpiecePreprocessTokenizer;
import org.tribuo.util.tokens.impl.wordpiece.WordpieceTokenizer;

import com.oracle.labs.mlrg.olcut.util.IOUtil;

public class WordpieceTokenizerTest extends TokenizerTestBase {

    @Disabled
    @Test
    void testSoftDash() throws Exception {
        String s = "\u00ad";
        System.out.println(s);
        System.out.println(WordpiecePreprocessTokenizer.isPunctuation(s.codePointAt(0)));
    }
    
    public static Stream<Arguments> testWordpiece() throws Exception {
        Wordpiece wordpiece = new Wordpiece("src/test/resources/co/huggingface/bert-base-uncased.txt");
        Tokenizer tokenizer = new WordpieceTokenizer(wordpiece, new WordpiecePreprocessTokenizer(), true, true, Collections.emptySet());

        return Stream.of(arguments(tokenizer, "", Collections.emptyList()),
                arguments(tokenizer, "test", Arrays.asList("test")),
                arguments(tokenizer, "partyg", Arrays.asList("party", "##g")),
                arguments(tokenizer, "whitecastleburgers", Arrays.asList("white", "##castle", "##burg", "##ers")),
                arguments(tokenizer, "hello world!", Arrays.asList("hello", "world", "!")),
                arguments(tokenizer, "prince humperdink", Arrays.asList("prince", "hum", "##per", "##din", "##k")),
                arguments(tokenizer, "prince४०८", Arrays.asList("[UNK]")),
                arguments(tokenizer, "the ४०८", Arrays.asList("the", "[UNK]")),
                arguments(tokenizer, "      ", Collections.emptyList()),
                arguments(tokenizer, "官", Arrays.asList("[UNK]")),
                arguments(tokenizer, "𧩙", Arrays.asList("[UNK]")),
                arguments(tokenizer, "官𧩙", Arrays.asList("[UNK]", "[UNK]")));
        
    }

    @ParameterizedTest
    @MethodSource
    public void testWordpiece(Tokenizer tokenizer, String text, List<String> expectedTokens) {
        List<String> actualTokens = tokenizer.split(text);
        Assertions.assertEquals(expectedTokens, actualTokens);
    }

    @Test
    public void regressionTest() throws Exception {
        Wordpiece wordpiece = new Wordpiece("src/test/resources/co/huggingface/bert-base-uncased.txt");
        Tokenizer tokenizer = new WordpieceTokenizer(wordpiece, new WordpiecePreprocessTokenizer(), true, true, Collections.emptySet());

        List<String> lines = IOUtil
                .getLines("src/test/resources/org/tribuo/util/tokens/impl/test/regression-text_bert-base-uncased.txt");

        final AtomicInteger progress = new AtomicInteger(0);
        assertAll(lines.stream().map(line -> {
            int p = progress.incrementAndGet();
            String[] data = line.split("\\t");
            String text = data[0];
            String[] expectedTokens = new String[data.length - 1];
            System.arraycopy(data, 1, expectedTokens, 0, data.length - 1);
            List<String> expectedTokensList = Arrays.asList(expectedTokens);
            List<String> actualTokensList = tokenizer.split(text);
            return () -> assertEquals(expectedTokensList, actualTokensList, "line="+p+": "+text);
        }));
    }
}