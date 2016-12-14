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
Here is a simplified visual representation of how I built my ensemble model. Starting with a set of about 50 raw fields, I feature-engineered a handful of potentially useful predictors, such as varsity sport involvement, winning an award, taking on a leadership position, etc. On the essay side, I employed NLP techniques to find the topic distribution of each essay (I’ll go into more depth about how this was done in the subsequent post). Additionally, I created a variable called word_sophistication, a proxy of how many ‘fancy’ words a student used in his/her essay (measured as total occurrence of sophisticated words / total word count). One might hypothesize that both extremes are negatively correlated with admissions outcomes: a value of zero might indicate a lack of wordsmanship, while a high value could point to a loquacious writer exorbitantly flamboyant in his lexical verbiage (excuse the irony). If so, the optimum must exist somewhere along this spectrum — we then let the beauty of machine learning take over to find this point/range.

### Evaluating the Model(s)
To evaluate the model, I spoke with the folks at AdmitSee and we agreed that precision would be the most fitting metric to evaluate a model’s success here. Models that prioritize precision tend to be more conservative in their probability estimates, which aligns quite well with AdmitSee’s goal of encouraging students to use their product even though they might be ‘star students’ to begin with.

A Receiver Operating Characteristic (ROC) curve illustrates the performance of a model as we vary the threshold at which we discriminate two classes. Basically, the goal is to maximize the area under the curve. In the graph below, we compare the performance of four models: i) Logistic Regression, ii) Random Forest, iii) a basic Ensemble Model [LR+RF], iv) a Grand Ensemble that builds on the basic Ensemble and combines it with a new model that incorporates essay features. Ignoring LR, while the area under the curves look visually indiscernible, the Grand Ensemble takes the cake, with a precision of 62.8 (compared to Ensemble’s 61.9 and RF’s 57.7).

### Interpreting the Model
Optimizing for precision is great, but what if we wanted to know how each variable affects your admissions chances? This is where Logistic Regression shines. In spite of its weaker performance, it is highly interpretable. More specifically, we can take the exponent of the coefficients to understand the marginal effect of each feature on the outcome variable.
Since I’m bound by an NDA, I can’t really disclose the details, but I can give a quick example. The coefficient for the binary variable leader is 0.82. Taking the exponent of that gives us 2.26. What that means is, if you aren’t already in a leadership position, taking one will more than double your odds of being admitted!

---

## Part 2: The Essay


---

## Final Thoughts & Caveats
Final Thoughts & Caveats
In our modeling phase, we implicitly assumed that these top schools apply the same criteria to vetting applicants every year, whereas in the reality, they probably update (even if slightly) what they look for in students as time passes. For next steps, I would focus on performing some more feature-engineering by looking at interaction effects (e.g. Varsity * Captain), and by exploring deeper effects intertwined across variables (e.g. a Hispanic student holding a leadership position in an Asian Student Society).
