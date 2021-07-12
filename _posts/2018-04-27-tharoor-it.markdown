---
layout: post
title: "Tharoor It"
description: "A fun application to calculate Tharoorian probability of some text."
date: 2018-04-27 12:00:00 +0530
image: '/images/tharoor_it/tharoor_it.png'
tags:   [tech, nlp]
---

I had recently started reading [An Era of Darkness: The British Empire in India](https://www.goodreads.com/book/show/32618967-an-era-of-darkness) and was dazzled by [Dr. Tharoor](http://www.shashitharoor.in/)'s wizardry of words. He calls himself an amateur historian while painting an honest picture of British Raj in India. I was mesmerized by his writing and wondered if any underlying pattern could be found using Deep Learning. Hackfest, our annual college hackathon was also approaching around the time, so I decided to execute a project related to Dr. Tharoor's literature in 36 hours. So when the contest started, we were trying to build a Tharoorian text classifier.

We planned to make a lightweight web application capable of taking a piece of text as input and estimating the probability of it being written by Dr. Tharoor. We were aware that it would be impossible to make a deployable model in just 36 hours. Nevertheless, we worked out a plan - Gather and process data, train a model, deploy on a local server and keep improving the model.

### Dataset
We collected 10 of Dr. Tharoor's published books, extracted ~30,000 sentences and labeled them as 1 while  manually removing short sentences consisting of only a few words. We randomly picked some books for negative samples and labeled ~200,000 sentences as 0. [100d GloVe embeddings](https://nlp.stanford.edu/projects/glove/) were used for creating the embedding matrix - vector representation of words in the dataset.

| Sentence        | Label           |
| ------------- |:-------------:|
| The British Raj is scarcely ancient history.  | 1 |
| The reason was simple: India was governed for the benefit of Britain.      | 1      |
| By breaking the law non-violently he showed up the injustice of the law. | 1      |
| This is not my fault. | 0 |
| Grown-ups never understand anything by themselves, and it is exhausting for children to have to explain over and over again. | 0 |
|  |  |
{: rules="groups"}


### Model
We decided to use a Bidirectional [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) for our purpose. Although we also tried 1D Convolutions which produced funny results. Training a bi-LSTM netowrk took 1 hour on our GPU. After our Keras model was trained, we exported it into a `.h5` file and used a flask server to [deploy it](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html) to a local server.

### Results
Although our model was a simple LSTM network without any attention, it gave decent predictions. Since the input to our network were single sentences only, it mostly depended upon presence of words to make predictions. Our training dataset only consisted of his books which kind of restricted our model's ability. Perhaps adding more data and using attention models would have given better results.

| Sentence        | Prediction(%)   |
| : ------------- |:-------------:|
| Earth is flat. | 49.42 |
| I will build a great, great wall on our southern border, and I will have Mexico pay for that wall. Mark my words.      | 84.38      |
| There was an idea to bring together a group of remarkable people, so when we needed them, they could fight the battles that we never could. | 96.64 |
| Exasperating farrago of distortions, misrepresentations & outright lies being broadcast by an unprincipled showman masquerading as a journalist.  | 87.03 |
| Albus Dumbledore didn't seem to realize that he had just arrived in a street where everything from his name to his boots was unwelcome.  | 0.05 |
| The sheep are running wild.  | 2.83 |
| History is boring.  | 90.49 |
| Physics is boring.  | 37.65 |
| I is extremely tired of reading history.  | 22.47 |

{: rules="groups"}

### Conclusion
Our team secured 6th rank out of ~50 participants. Along with this project, we also attempted Samsung's Bixby challenge based on NLP and were 2nd runner-up.