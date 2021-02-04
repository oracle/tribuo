package org.tribuo.interop.onnx.bert;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

public class BERTFeatureExtractorTest {

    @Test
    public void testTokenizerLoading() throws URISyntaxException, IOException {
        Path vocabPath = Paths.get(BERTFeatureExtractorTest.class.getResource("bert-base-cased-vocab.txt").toURI());
        Path tokenizerPath = Paths.get(BERTFeatureExtractorTest.class.getResource("bert-base-cased-tokenizer.json").toURI());
        BERTFeatureExtractor.TokenizerConfig config = BERTFeatureExtractor.loadTokenizer(tokenizerPath);

        List<String> vocabList = Files.readAllLines(vocabPath, StandardCharsets.UTF_8);

        Assertions.assertEquals(config.tokenIDs.size(),vocabList.size());

        for (String vocabElement : vocabList) {
            Assertions.assertTrue(config.tokenIDs.containsKey(vocabElement));
        }

        Assertions.assertEquals(100, config.maxInputCharsPerWord);
        Assertions.assertEquals(false, config.lowercase);
        Assertions.assertEquals(false, config.lowercase);
        Assertions.assertEquals("[UNK]",config.unknownToken);
        Assertions.assertEquals("[CLS]",config.classificationToken);
        Assertions.assertEquals("[SEP]",config.separatorToken);
    }

}
