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

this sort of an input output mapping.
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


