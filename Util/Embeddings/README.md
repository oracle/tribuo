# tribuo-util-embeddings

Java code for running ONNX format embedding models.

## ONNX model export from HuggingFace Transformers

First install HF Transformers and Optimum into your Python venv:

```
pip install transformers optimum[onnxruntime] sentence-transformers
```

Then run the following at the python REPL:

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_checkpoint = "<model-name>"
save_directory = model_checkpoint + "-onnx"
ort_model = ORTModelForFeatureExtraction.from_pretrained(model_checkpoint, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
```

For example, to download the sentence transformers version of MiniLM-L6-v2 use `sentence-transformers/all-MiniLM-L6-v2` as `model_checkpoint`.

## Running a simple embedding

The following command loads in a MiniLM prepared in the above way, then runs inference on each line of the text file `test.txt`
generating a json array output in `minilm-output.json`.

> mvn exec:run --args="-c configs/minilm-config.xml -e minilm -i test.txt -o minilm-output.json"
