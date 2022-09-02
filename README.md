
# Image Caption Generation Using CNN and LSTM

# 1 Motivation
Image caption generation is a task that requires proper understanding of both Computer Vi-
sion(CV) techniques and Natural Language Processing(NLP) methods. It uses techniques from
both domains and puts them together hand in hand to generate captions that are apt for any
given image. It holds great importance and value in some of the domains in the real world today.
The following areas make use of this technique as a part of their pipelines -
1. Self-Driving cars: One of the thriving industries today is the automobile industry and their
biggest challenge now is to be able to make Advanced Driver Assistance Systems(ADAS) that
aid the driver with information about the environment around the car. This task renders
itself useful to such a use case as we can take an image of the surroundings and if it can
automatically generate a useful caption from the image, it can then be converted into a voice
command or information that is useful to the driver.
2. Google Image search: Another use case is to be able to generate a useful caption for
any given image initially before searching in Google Images. This way, once the caption is
generated, we could use that caption to search on Google for better and apt images.
3. Aid to help the blind: NVIDIA is working on a project that obtains image from the
surrounding and converts the image of surroundings to textual information that can be played
on a earpiece to the blind helping them understand their surroundings in more depth.
The application we have tried to bring to light in this project is (b) Google Image Search.
We wanted to understand and learn about the model that can do the image caption generation and
train one ourselves to understand the various nuances involved in it. We have implemented the use
case by trying to make use of \Web Scraping", a technology that is often used to scrape data
from the browsers to find matching apt images/text/documents etc. Thus this project implements
the Image caption generation using CNN and LSTM two deep learning models for the computer
vision and natural language processing parts respectively.

### Introduction
The idea of caption generation using natural language was first done by Andrej Karpathy, Li
Fei-Fei in 2015's IEEE Conference on Computer Vision and Pattern Recognition (CVPR). His
conference paper[3] was revolutionary in the domain of CV and NLP at its time and has hence
been used across various domains for various use cases as mentioned in the motivation section
above. The entire project was worked using Google Colab's python environment using Colab's
TPU processor. The models's layers and training, loss functions have all made use of Keras
library's functions. The rest of the document is ordered as follows - Problem Statement, Datasets,
Implementation, Results, Conclusion, References.
Figure 1: Basic idea of combined model
* Photo Feature Extractor. This is a 16-layer VGG model pre-trained on the ImageNet dataset.
We have pre-processed the photos with the VGG model (without the output layer) and will
use the extracted features predicted by this model as input. Sequence Processor. This is a
word embedding layer for handling the text input, followed by a Long Short-Term Memory
(LSTM) recurrent neural network layer. Decoder (for lack of a better name). Both the
feature extractor and sequence processor output a fixed-length vector. These are merged
together and processed by a Dense layer to make a final prediction. The Photo Feature
Extractor model expects input photo features to be a vector of 4,096 elements. These are
processed by a Dense layer to produce a 256 element representation of the photo.
* The Sequence Processor model expects input sequences with a pre-defined length (34 words)
which are fed into an Embedding layer that uses a mask to ignore padded values. This is
followed by an LSTM layer with 256 memory units.
Both the input models produce a 256 element vector. Further, both input models use
regularization in the form of 50
* The Decoder model merges the vectors from both input models using an addition operation.
This is then fed to a Dense 256 neuron layer and then to a final output Dense layer that makes
a softmax prediction over the entire output vocabulary for the next word in the sequence.
*. Problem Statement
The problem statement for this project is to be able to implement a unified model that has a
part working with the computer vision side of the project and another model that works with
the natural language processing side. Thus we have made use of two different network models -
Convolutional Neural Network (CNN) for the CV aspects and a Long Short-Term Memory(LSTM)
for the NLP aspects. Finally the two models are combined into one to link the NLP outputs with
its corresponding CV counterparts, thereby producing a caption for the given image. For training
2
this model, we have made use of the Flickr8k dataset. The details of the dataset are explained in
the next section.
4. Datasets
## Training Data
The dataset used in this project is similar to the one used in - the Flickr8K dataset. The
Flickr8K dataset is a free and readily available dataset which we have taken from this link. This
dataset consists of two parts:
* Flickr8K Dataset: contains a total of 8092 images in JPEG format with different shapes and
sizes. Of which 6000 are used for training, 1000 for test and 1000 for development. This
covers the number of images that are there in this.
* Flickr8K text: Contains text files describing train set, test set. Flickr8k.token.txt contains
5 captions for each image i.e. total 40460(8092*5) captions.
The following is a sample of the image and its corresponding 5 caption texts in the dataset.
(a) Captions for the girl's image
(b) The girl's image
Figure 2: Captions and Features for the above girl's image.
## Testing data
For testing we have implemented Web Scraping. Web Scraping is a technique employed to
extract large amounts of data from websites whereby the data is extracted and saved to a local
le in your computer. We have implemented a web scraping function in our code where we enter
a query term that describes a component in the image we want to search for on the internet. This
image is then scraped of the internet and directly fed to the testing part of the model to obtain
the captions.
3
###  Implementation
##  Photo Preparation
The idea in this project is to extract the features of the image and directly correlate the higher
level features with the appropriate text words that get generated in the NLP network. Thus in
order to be able to extract features from the image we use a transfer learning model with VGG
as the trained transfer model. Instead of running our images through the entire architecture, we
shall download the model weights and feed them to the model as an interpretation of the photo
in the dataset. Keras provides us with VGG class which enables us to do just the same. Here
we freeze all the layers and their weights except the last two layers and pass our images through
them to learn the features thoroughly.
Figure 3: Transfer learning model
We load each photo and extract features from the VGG16 model as a 1x4096 vector. The
function will return a dictionary of image identifying tags(Image names on the 
ickr8k dataset),
and the corresponding feature vectors. The following is a snippet that indicates a feature and
image id generated from the code.
Figure 4: Feature vector and its corresponding image id from VGG
##  Text Preparation
The model makes use of a LSTM as mentioned before which is a Recurrent Neural Network(RNN)
for the NLP activites. A RNN is a class of artificial neural networks where connections between
nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic
behavior. Derived from feed-forward neural networks, RNNs can use their internal state (memory)
to process variable length sequences of inputs. This makes them ideal to be able to generate words
iteratively to form a sentence/caption in our case for our image.
Thus in order to prepare the given text in our dataset to meet the RNN standards, we have
to perform two main pre-processing steps:
1. Tokenization: Tokenization is a way of separating a piece of text into smaller units called
tokens. Here, tokens can be either words, characters, or subwords. Hence, tokenization
can be broadly classified into 3 types { word, character, and subword (n-gram characters)
tokenization. Thus for example if we have the sentence " A cat is chasing the mouse.",
##
the tokenization's output will be the following tokens: A, cat, is, chasing, the, mouse. It
splits the sentence at every delimiter(here, space). For our given task the following steps are
performed as part of the tokenization process:
* Separate token Id and Image descriptions word by word and put them into two separate
variables.
* Remove the file extension from the image ID.
* Now concatenate all the word of a single caption into a string again.
* For every image ID store all 5 captions.
* Return as a dictionary consisting of lists of image IDs mapped to their corresponding
captions.
2. Vocabulary: The main aim of Tokenization in any NLP task is to ultimately end up
with the suitable vocabulary to train the model. Likewise, in order to create meaningful
vocabulary for our model to learn from we further do the following pre-processing steps to
our tokens:
* We convert all the words to Lower case.
* Remove all the punctuation.
* Remove " 's " and 'a'
* Remove all word with numbers in them.
Finally, our code returns the cleaned words as a set named so that we have unique items in
our vocabulary list which was extracted from the annotations document.
Now, we make a dictionary of image identifiers and descriptions to a new file and save the
mappings to a file. This is later directly called for any further usage. At the end of this step of
text preparation we have a vocabulary size of 8793 from the 8092 images' corresponding captions.
The following is the output after the text has been pre-processed for the following image id:
"1000268201 693b08cb0e".
Figure 5: Captions after text preparation
##.  Creating the final model from CNN and LSTM
The next step is to combine the CV and NLP elements into a single model. The following function
performs the transfer learning step for VGG and the NLP as the two inputs to the final model and
lastly inserts the last layers(two Dropout layers and a Dense layer) before adding the LSTM. The
following is the function snippet that performs the functionality of piecing together the model.

The model summary as produced by the function is as follows:

Having done this model, the training was then done for 20 epochs. After training on the 6000
images from the dataset,we move onto testing.
## Testing
The metric used to test this project is the Bilingual Evaluation Understudy Score(BLEU).
This score is a metric for evaluating a generated sentence to a reference sentence. A perfect match
results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0. Thus we first tested
it using 1000 images from the dataset Flickr8K and obtained the following BLEU values:
6

## Web Scraping
As mentioned before, we tried this image caption generation with web scraping to cater to the
application of using this tech for Google Image searching. The following function is an implemen-
tation of the web scraping:

The "query" function paramater bears the string we want to search and scrap o of the web.
This is given in the fucntion call of this function during the testing. Following is a snippet of the
function call.
## Results
For the above testing functiomn call, we obtained the following image o the web and the generated
caption:

##  References
[2]https://blogs.nvidia.com/blog/2018/04/26/deep-learning-app-seeing-ai-app/
[3]A. Karpathy and L. Fei-Fei, "Deep visual-semantic alignments for generating image descrip-
tions," 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston,
MA, 2015, pp. 3128-3137, doi: 10.1109/CVPR.2015.7298932.
8
