# ML-project



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

#### 3. Hybrid
Somehow do both of the above.
