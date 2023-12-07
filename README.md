# Recommender System Using Image.

## Idea
 The input in ImageNet VGG16 model passes through a set of convolution, pooling, and fully connected layers to the last softmax layer for the final classification task. The penultimate layer before softmax captures all image information in a vector such that it can be used to classify the image correctly. So, we can use the penultimate layer value of a pre-trained model as our image embedding.
 We can utilize this image embedding with user-ratings to recommend users.