# Intro-to-Machine-Learning


So this course will cover model based techniques for
extracting information from data.
With some sort of an end task in mind.
So these tasks could include, for example,
predicting an unknown output given some corresponding input.
We could simply wanna uncover the information underlying our data set with
the goal of better understanding what's contained in our data that we have.
Or we can do things like data driven recommendation,
grouping, classification, ranking, et cetera.

Types of learning:

1. Supervised 
2. Unsupervised

### Supervised Learning

**What do we mean when we say we want to perform supervised learning?**

In a nutshell, what we're saying is that we want to take inputs and
predict corresponding outputs.

So for example, if we wanna do regression, we might have
pairs of data, in this case a one dimensional value for
x and a corresponding one dimensional value for t.
And then we would want to learn some sort of a function so that we input x and
we make a prediction for the output t.

![capture 1](./img/capture1.png)

So for example, here we have several data points as
indicated by circles where the x axis is the input for that particular point.
And the t axis is a corresponding output.
And now that we have this data set of these several points
we want to define some sort of a regression function.
For example, this blue line that in some way interpolates
the outputs as a function of the inputs.

And then the goal is, given this smoothing function that we've learned, for
some new input x, we want to predict the corresponding output t.
So for future time points we obtain x,
we obtain an input, and we wanna predict the corresponding output.
So that's regression.
We say it's regression because the outputs are assumed to be real valued.

___

Classification is another supervised learning problem that is slightly
different.
The form, or the structure, is very similar.
We have pairs of inputs and outputs and we get this data set,
which has many of these pairs of inputs and outputs.
And we want to learn some functions so
that in the future when we get a new input for which we don't have the output,
we can make a prediction of the output that's going to be accurate.

However the key difference here is that where with regression the output is a real
valued output, with classification it's a discrete valued thing.
So it's a category or a class.

![capture 1](./img/capture1.png)

So in this right plot, what I'm showing are input output pairs,
except now the input is a two dimensional vector.
So here the input would be this two dimensional point and
the output for this plot is being encoded by a color.
So the output could be one of two values.
Either a blue value or an orange value.
So in this case we want to take our data inputs and
classify them into one of two outputs.

So we get a data set like this, with all of these input locations and
the corresponding color coded output.
And now our goal is to learn some sort of a function, a classifier, so
that we can partition the space, such as is shown here.
Where for a brand new point, any of these points that we don't have the output,
we can evaluate the function at that point and make a prediction of the output.

So we might say, for this data set, we would partition this entire region
here, these two regions into the orange class.
And this region here into the blue class.
So any new points falling in this region will be assigned to the blue class.

So the key here with supervised learning is that we're learning
an algorithm based on a function mapping from input to output.
The outputs are basically going to be telling us how to map the inputs so
that we have an accurate function.

![capture 2](./img/capture2.png)


So to look at a classic example we could think of spam detection.
Given some set of inputs, like these two chunks of text, we would want to
assign it a label, plus one, or minus one, sometimes we would say plus one or zero.
But we would want to assign it one of two possible labels.
One label would correspond to an email that is spam and
we would want to then automatically delete that email.
And the other class would be non-spam emails.
Emails that we want to put into our inbox and actually read.
So it's essentially a filtering problem.
So, for example, we might have a data point like this, containing this text,
and we would want to now input this into some sort of a function and
say is it spam or not.
In this case, most likely it's not.
Or a data point like this, this piece of text,
where we would input into the same exact function with the same classifier.
And in this case that same classifier would say this email is a spam.
So we classify this email to spam and this email to non-spam using the same
classifier, learned from examples of labeled spam and labeled non spam emails.

### Unsupervised Learning

Supervised learning is very nice because we know
that we want to map an input to an output.
And honestly we don't necessarily even care how it's mapped.
We simply want to say here's my input, what should I map It to as an output.
And we measure the performance based purely on how well it does that task.

This is sort of an input output mapping.
We want to perform more abstract or
vague tasks such as understanding what is the information in our data set.
For example, we don't have an infinite amount of time to read so
many thousands or millions of documents so we want a fast algorithmic way for
taking in information, taking in data and extracting the information for us.

So, for example, with unsupervised learning we might wanna do something like
topic modeling, where we have many texts,
data, many documents provided to us.
We don't have any labels for these documents, all we have is the text for
each document.
And then we want to extract what is the underlying,
what are the underlying themes within these documents.
So that's the idea of topic modeling.

We also might want to do recommendations.
This would be where we have many users and many objects.
And the users will give feedback or input on a subset of these objects.
Either through a review or through some sort of a rating, like a star rating.
For example with Netflix, a user could rate a movie one to five stars.
And we want to take all of this information and
learn some sort of a latent space where we can embed users and
movies such that users who are close to each other share similar tastes.
Movies that are close to users are somehow appropriate recommendations to be
made to those users.
Movies that are close to each other are similar in their content and
things that are very far apart are very unlike each other.
So we wanna learn this information simply from the data.
From the raw data and some model assumption that we have to make.

## Data Modelling

Everything that we are going to discuss in this course
can in some way boil down to what's the flow chart that's contained on this slide.
So we have four components here.
We have the data, which I'm calling block one.
We have a model for the data which I'm calling block two.
We then have some sort of an inference algorithm, block three.
And we have a task or some goal that we want to do like predicting or
exploring the data which I'll call block four.
And they all fall into, all the algorithms or
techniques, the motivations have this same sort of flow.
So we have a data set, we start with that.
We have a goal, what we want to do with this data set.
We have this here.
We then have to define some sort of a model that's going to get us to our goal.
This part here, we have to define a model for the data that's going to do for us
what we wanted to do like make predictions or find the structure in the data.
But this model is going to have unknown parameters to it and
we need to learn these parameters.
And so that's where block three the inference portion comes into play.
So we basically say here's a model with some unknown parameters.
Here's a dataset that we're going to say that we want to model with this model.
Now tell me,
what are the parameters I should set the model to in order to model it well?
That's block three.
And when we learn those parameters we can then take new data and make predictions or
explore the data set.

![capture 3](./img/capture3.png)

So for example, the difference between supervised versus unsupervised learning,
can be thought of as comparing as block one and block four.
I'm thinking of block one and block four.
So what is the data and what do we want to do?
If we have data where we want to predict the output for an input by learning
some function that maps inputs to outputs, then we're doing supervised learnings.
So that's what block one and block four will tell us to do.
If we want to simply learn the structure underlying our
data set with some sort of a model.
For example, we have documents and we want to learn the topics of those documents to
explore the data set, we might want to do an unsupervised learning model.
And so unsupervised versus supervised can be thought of as what block one and
block four is telling us and then block two and
block three are the set of techniques that we perform toward that end.

If we think probabilistic verses non-probabilistic models we're now not
really thinking about the data or what we want to do with the data.
We're thinking about the model itself primarily.
So we're defining a model that uses probability distributions in
the probabilistic case.
Or we're defining a model where the probability distributions really don't
come into play at all in the no probabilistic case.
And also to a certain extent the probabilistic versus
non-probabilistic dichotomy appears in block three as well.
In that some algorithmic techniques are purely motivated by fact that we
are doing a probabilistic model.
However, a lot of techniques for non-probabilistic models
apply equally as well to probabilistic models with no modifications at all.
So to some extent, block three is shared across the two, and
the difference between probabilistic and non-probabilistic modeling is
primarily in the types of models that we're going to define.

### Gaussian Distribution (Multivariate)

![capture 4](./img/capture4.png)


**Block 1**:
We assume that we have data points x1 through xn.
We have n observations, and each observation is a d
dimensional vector that can take any value.
So this is the data that we're dealing with.
There's no other data that we have that we would want to model, we have a data set
of n three dimensional vectors and we want to model those vectors.
We believe those vectors can take any value.

**Block 2**:
Block 2 then is the model that we wanted to find.
So we have this data, now what do we wanna do with it?
How do we wanna model it I should say.
So for this problem, what we're going to think of, and
I'll return to these in more detail in the rest of the lecture,
we're going to say that we're gonna model this data with.
A Gaussian distribution, a multivariate Gaussian probability distribution.
And we're going to say that the n datapoints are IID, independent and
identically distributed according to this distribution.
So we'll return to that in a few moments but that's block 2.
We've now defined a model for the data.
We've said this is the structure that we wanna learn from the data because this
is a probability distribution, so we're looking at this problem probabilistically.
We now get to add the intuition that we're saying this is how the data was generated.
We have a probability model, that's saying the data was generated in this way.
Okay, so we have the data, we've defined the model for the data.

**Block 3**:
Now we have to transition the block 3 and learn the parameters of the model.
So, a Gaussian has parameters, we'll discuss it since it's on the slide.
We have the data that we say came from this Gaussian.
How do we now fix or
tune the parameters to this Gaussian in a way to explain the data?
And so this is going to lead to a problem of having to define
some sort of an objective function.
And then defining a way for optimizing that objective function.
And so what we'll discuss in the rest of this lecture is something called
maximum likelihood.

**Block 4**:
And block four we can leave undefined for
now, we don't need to Say what we want to do with this data.
We don't need to say why it is that we wanna learn a model for the data.
It could simply be that we just want to reduce the data so
that we can have a compact representation of it and
then throw the data away, or many other reasons.

___

Okay so, looking at block two,
we're going to define a multivariate Gaussian distribution.
So to refresh our memories, when we define a probability
distribution we're saying that the probability of x or the density of x
Given parameters mu and sigma, is equal to some function.
That function has to be non negative everywhere.
So it's a function of x as well as mu and sigma.
It has to be non negative everywhere and it has to integrate to one.

And the multi variate Gaussian has this Specific form where we take the input
vector, so this is a d dimensional vector, we subtract off it's mean,
we perform this quadratic function using a mean vector mu, that's d dimensional.
And D by D matrix sigma, that's positive definite.
We evaluate this function, and that gives us a density for
a particular point x, using a galcene with mean mu and covariant sigma.
So we get something that looks like this.
If d is 2, so we have a 2-dimensional vector x,
we're evaluating this function p in a 2-dimensional space.
This 3rd dimension is then the value of the function at a particular location.
So would assume, looking at this function, that the mean is somewhere around (0,0).
And the covariance defines this sort of a shape.
So a point in this region will have very tiny density, very small density.
So we won't expect to see much data in this region.
Whereas in this hump region we have large density,
meaning most of the data is gonna come from this region.


## Block 2: A Probabilistic Model

Again, in this case we're defining a probabilistic model.
So we're thinking probabilistically here.
So what a probabilistic model is
is simply a set of probability distributions on our data.

So we have data x.
We have to define some probability distribution on x.
That distribution is going to take parameters theta, which we don't know.

![capture 5](./img/capture5.png)


So by this definition, we haven't actually defined the final distribution,
we've defined a distribution family.
Meaning, this probability distribution can take many different values for theta.
And no matter what value we set theta to,
we're always working within the same distribution family.
So in the case of the Gaussian distribution that we've been thinking of,
this density, p of x given theta, is a multivariate Gaussian.
That's the distribution family, a multivariate Gaussian.
And it takes parameters mu and sigma, mean and covariance.
And for all values of mu and sigma we're always working with the same density,
the same multivariate Gaussian, the same distribution family.

So we've defined this model, we've defined the distribution family.
We've also made an assumption, an IID assumption.
What IID stands for is Independent and Identically Distributed.
And so this is a very common assumption that makes working with data much easier.

What this means is that, intuitively, it means that every single data point is
simply independent from every other and it has the same distribution family.
So if we have n multivariate Gaussians,
we assume that every single one of those n vectors in our data set was generated
from a multivariate Gaussian using exactly the same mean and covariance parameter.
And they were generated without reference to what any of the other values were.
It was essentially a brand new experiment for each of the n vectors that we obtain.

You can also think of this in coin flipping terms,
where if we flip a coin where the coin has a 0.5 probability of heads, 0.5 of tails.
The density, or the distribution in this case, not density that we're working with,
is a binomial distribution with parameter 0.5.
So that's the distribution and the independent and
identically distributed comes into play by saying that every coin
that we flip has the same distribution of heads versus tails.
And they're all flipped completely separate of each other.
They don't influence each other in any way.

So we have a probability of n data points, n d-dimensional vectors,
coming from a distribution with parameter theta.
And we can say that that joint density, the joint probability of all n of these
observations is simply the product of the probability of each of them individually.
So again, if these weren't independent, if these x values were not
independent of each other, we could not write it as this product form.

![capture 6](./img/capture6.png)


