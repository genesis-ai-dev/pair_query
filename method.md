Say we are trying to translate a sentence into German, but needed a way to evaluate how good the translation is.

Assuming we have a corpus of aligned translation pairs, we can do this even without knowing the ground truth.

Here's what we do:
- we take our sentence in both languages, our english and our draft translation. Then, we can randomly pick N examples from the corpus.
- we can then measure how similar our english sentece is to the reference english sentence. If we have three examples, than our english similarity matrix might look like this [.3, .1, .2]  
- we can then do the same for our german sentence, and since pairs are aligned, our german similarity matrix might look like this [.4, .2, .3]

Even though the two matricies are different, we can see that they have a high correlation!

This is a good sign that the translation is good!



