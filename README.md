# ML-project

## Results
For results, we refer to the [paper](./neural-data-compression.pdf).

## Idea

Suppose we have a text with a fixed alphabet (like DNA sequences). We can compress it as following:
- First we translate this text to a better compressable format. We do this as following:
  We have a trained model that given a sequence of characters then tries to predict the next character. So for a fixed alphabet we loop over each character in the input string and calculate the probability of each character given the previous characters. We then replace the character (that we want to predict) with a number that represents the nth most probable character. So 0 if the model predicted the next character correctly, 1 if the model predicted the second most probable character and so on. We (hopefuly) end with a sequences of numbers that is more compressable than the original text because there should be alot more 0's then 1's etc.
- Next we can use arithmetic encoding to compress the sequence of numbers.


Decompressing is simply the inverse operation.

## Model
There are multiple ways to create a model:
#### 1. Static
The model is trained on a large dataset and then used to compress any text with the same alphabet.

#### 2. Dynamic
No model is trained prior to compressing the text. The model is trained on the text that is going to be compressed. 

#### 3. Adaptive
The model is trained while compressing. 

## How to run

A script is provided to run the static model on the HPC. This is the recommended way, as running it locally is possible but small edits in the code are needed.

To run the static model on the HPC:

First the `install_venv.pbs` script should be run:
```qsub install_venv -v cluster=<CLUSTER_NAME>```

Then, the specified cluster should be changed in the `run_static.pbs` script on line 10. After that `run_static.pbs` can be run like this:
```qsub run_static.pbs```

`run_static.pbs` simply runs StaticCompressor.py. If you want to run `StaticCompressor.py` locally, this is its syntax:

```python StaticCompressor.py /path/to/dataset.txt [model_type]```

with model type being either `english` or `dna`. If the DNA model is used, the hyperparameters should manually be changed in `train.py` on line 31.

NOTE: the model will only train if a file named `char_model.keras` doesn't exist in the `VSC_DATA` directory.


To run the adaptive method:
```python
python ./src/AdaptiveCompressor.py 
```
To run the dynamic method:
```python
python ./src/DynamicCompressor.py 
```

