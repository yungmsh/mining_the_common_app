# Mining the Common App

### Introduction
The industry built around college admissions is a pretty saturated one — from private ‘elite’ tutors in Asia, to global test-prep powerhouses like Kaplan, and even to TaskRabbit-esque marketplaces for ‘essay consultation services’ (read: latent plagiarism). Yet it seems tutors, teachers, and consultants alike have yet to crack the code on admissions. Does a magic formula even exist? Probably not, but there are likely decision rules within admissions committees we don’t know much about, and there are most certainly underlying relationships across acceptances and denials that we can learn from.

For my final capstone project at Galvanize, I had the fortune of working with AdmitSee, an online college application resource where prospective students can browse the profiles and essays of real admitted students. With this unique data, I set out to answer two key questions:

1. Can we create a better predictive model than the existing probability-calculators for college admissions?
2. What insights can we glean from the Common App essay, both on an individual and an aggregate level?

## Part 1: The Model

### Constraining the Problem
After doing a train-test split on my dataset, I had about 12k students to work with. Among these, even the schools with the most data were several hundred at most. Since modeling at a school level would be potentially quite inaccurate, I chose to constrain the problem to predict a single binary outcome: admission into a ‘top school’. In other words, if a student was accepted into any of the ‘top schools’ (defined as Ivies, Stanford and MIT) the outcome variable is 1, and 0 otherwise.

### Building an Ensemble Model
Here is a simplified visual representation of how I built my ensemble model.
![alt tag](https://raw.githubusercontent.com/yungmsh/capstone_project/tree/master/imgs/model_pipeline.png)
Starting with a set of about 50 raw fields, I feature-engineered a handful of potentially useful predictors, such as varsity sport involvement, winning an award, taking on a leadership position, etc. On the essay side, I employed NLP techniques to find the topic distribution of each essay (I’ll go into more depth about how this was done in the subsequent post). Additionally, I created a variable called word_sophistication, a proxy of how many ‘fancy’ words a student used in his/her essay (measured as total occurrence of sophisticated words / total word count). One might hypothesize that both extremes are negatively correlated with admissions outcomes: a value of zero might indicate a lack of wordsmanship, while a high value could point to a loquacious writer exorbitantly flamboyant in his lexical verbiage (excuse the irony). If so, the optimum must exist somewhere along this spectrum — we then let the beauty of machine learning take over to find this point/range.


### Evaluating the Model(s)
To evaluate the model, I spoke with the folks at AdmitSee and we agreed that precision would be the most fitting metric to evaluate a model’s success here. Models that prioritize precision tend to be more conservative in their probability estimates, which aligns quite well with AdmitSee’s goal of encouraging students to use their product even though they might be ‘star students’ to begin with.
<br>
A Receiver Operating Characteristic (ROC) curve illustrates the performance of a model as we vary the threshold at which we discriminate two classes. Basically, the goal is to maximize the area under the curve. In the graph below, we compare the performance of four models: i) Logistic Regression, ii) Random Forest, iii) a basic Ensemble Model [LR+RF], iv) a Grand Ensemble that builds on the basic Ensemble and combines it with a new model that incorporates essay features. Ignoring LR, while the area under the curves look visually indiscernible, the Grand Ensemble takes the cake, with a precision of 62.8 (compared to Ensemble’s 61.9 and RF’s 57.7).

### Interpreting the Model
Optimizing for precision is great, but what if we wanted to know how each variable affects your admissions chances? This is where Logistic Regression shines. In spite of its weaker performance, it is highly interpretable. More specifically, we can take the exponent of the coefficients to understand the marginal effect of each feature on the outcome variable.
<br>
Since I’m bound by an NDA, I can’t really disclose the details, but I can give a quick example. The coefficient for the binary variable leader is 0.82. Taking the exponent of that gives us 2.26. What that means is, if you aren’t already in a leadership position, taking one will more than double your odds of being admitted!

---

## Part 2: The Essay
As a quick refresher, Part 1 discussed the process of building a classifier that predicts a student’s probability of being admitted into a ‘top school’. Part 2 addresses the following: <i>What insights can we glean from the Common App essay, on both an individual and an aggregate level?</i>

### Stop Words & Lemmatizing
Before doing any analysis, I first removed the standard ‘stop words’, e.g. ‘you’, ‘me’, ‘him’, ‘of’, ‘the’, … words we use in our everyday life that (in most cases) contain basically no useful information for text analysis. Next, I needed to reduce words to their basic root form. Plurality should be singularized, adverbs reduced to basic adjectives, among other things. There are generally two ways to go about this: stemming and lemmatizing. Stemmers tend to be more ‘strict’ (i.e. chops off more of a word), and the resulting tokens are often not real words, e.g. familial → famili; arguing → argu. Lemmatizers are more ‘lenient (i.e. doesn’t reduce the word as much), and the resulting tokens are often real, interpretable words. For my dataset, I first used NodeBox’s Linguistics library to convert all verbs into present tense (NLTK isn’t great with tense-correction), and then used NLTK’s SnowballStemmer to stem words. I chose to use a stemmer over a lemmatizer because the preservation of real words was not as important to me as grouping words with similar roots together.

### Vectorizing the Essays (using TF-IDF)
Next, we want to represent each essay as a numerical vector, so that we can make calculations and comparisons with other essays later on. We do this by constructing an enormous matrix where the rows are essays and the columns are words (the aggregate column space is basically the entire vocabulary across all essays). Each cell represents the ‘importance’ of a particular word in a given essay. For instance, cell (247, 1928) refers to the importance of word 247 in essay 1928. Note that the ordering of words and essays have no meaning here. The ‘importance’ value can be a simple word count, but a more robust approach uses something called ‘term frequency - inverse document frequency’ (TF-IDF). This basically computes a normalized word count for each essay, but weighs each word inversely proportional to the occurrence of that word in the entire corpus. Picture an essay in your head. If you see a word that’s also widely used in every other essay, that word should be given less importance. On the other hand, if you see a word that’s unique to that essay you’re reading (i.e. rare across all essays), it should be given higher importance.

### Topic Modeling (using NMF)
After vectorizing the essays, our matrix takes the form of the square grey box on the left in the graphic below (note: in reality it is not square-shaped at all — in our case the number of words far exceed the number of essays).

<!-- ![alt tag](https://cdn-images-1.medium.com/max/800/1*kZy81Ogwt-A17ZfodN9CQg.png) -->
![alt tag](https://raw.githubusercontent.com/yungmsh/capstone_project/tree/master/imgs/img1.png)

It’s great that we’ve represented the essay as vectors, but in reality the matrix is highly sparse (i.e. mostly filled with zeros) so it’s still a little difficult to make meaningful calculations. Enter dimensionality reduction. The idea is to reduce the number of dimensions by some order so that we can more easily perform operations on the data. On the plus side, it helps reduce overfitting, lowers computational costs, and prevents the ‘curse of dimensionality’. On the flip side, we forgo some information as we are essentially ‘throwing out’ columns of data. There are many techniques to do this — in this case, I chose to use non-negative matrix factorization (NMF). The basic premise of NMF is to deconstruct your original matrix into two separate matrices: a ‘long’ one and a ‘fat’ one. When you multiply the two together, you get a reconstructed matrix that is approximately equal to your original matrix. It’s called ‘non-negative’ because we don’t allow any negative values, giving the benefit of interpretability especially in the context of text analysis and ratings/reviews data (e.g. Netflix, Yelp). In our case, the ‘long’ one is our new dimensionality-reduced matrix, and the ‘fat’ one is a reference guide that contains semantic information (see below).

### Essay Topic Distribution
Why was all that important? Well, I built a tool for AdmitSee where you can upload your essay, and it will tell you the topic distribution of your essay, according to the seven topics the NMF algorithm learned. In addition, it also shows you the three most ‘similar’ essays to your essay. How is similarity calculated? You can choose from two options: Euclidean distance of the topic distribution, or Cosine similarity of the TF-IDF essay vector.

---

## Final Thoughts & Caveats
As exciting and rewarding as this project was, there is always room for improvement. In our modeling phase, we implicitly assumed that these top schools apply the same criteria to vetting applicants every year, whereas in the reality, they probably update (even if slightly) what they look for in students as time passes. For next steps, I would focus on performing some more feature-engineering by looking at interaction effects (e.g. Varsity * Captain), and by exploring deeper effects intertwined across variables (e.g. a Hispanic student holding a leadership position in an Asian Student Society).
<br>
To take this project to the next level, I would also explore using Latent Dirichlet Allocation (LDA) for the topic modeling portion. LDA is a generative statistical model that assumes every essay has an underlying distribution of topics, and every topic has an underlying distribution of words. Recent literature has suggested that this probabilistic approach can yield better results, so it would be a natural next step to explore.
<br>
As AdmitSee continues to grow and collect more data, it would be interesting to see how the visualization of schools above differs between undergraduate and graduate essays, and to also look at trends over time (e.g. have certain schools shifted from more career-driven to more personality-driven essays?)
