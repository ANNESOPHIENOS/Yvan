
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from pylab import contour
import random
np.random.seed(10)
get_ipython().magic('matplotlib inline')


# In[2]:


data_path = './'
train = pd.read_csv(data_path + 'classification.csv').fillna(0.0)


# ## 1. Simple Linear Regression using scikit-learn
# 
# You are doing some work for a large British retailer; Ocset.
# 
# Ocset are planning to send a coupon for pasta to customers. They have asked you to determine which customers are most likely to redeem that coupon, so that they can target the right customers.
# 
# You have told Ocset that you'd like to build a regression model on historic redemption data. In particular, you have told them that you'll use historic behavioural data (e.g. pasta sauce coupon redemptions) as your $X$ to predict historic coupon redemptions on previous pasta coupons (i.e. your target variable $y$). 
# 
# Ocset currently use 'pasta sauce' redemptions to determine who receives pasta coupons (pasta and pasta sauce coupons occur sequentially; never in the same mailer). Those that have purchased the most pasta sauce coupons in the most recent mailer are targeted for pasta. 
# 
# You have told your client that you will use simple linear regression to test the claim that pasta sauce coupon redemptions is an effective method for targeting pasta coupons.

# Firstly, let's examine the training set...
# 
# 
# ##### Q 1.1: Describe the 'train' object. What type of object is it? How many rows and columns? What types exist in this object?

# In[6]:


train


# In[7]:


train.describe()


# In[8]:


print train.dtypes


# We need to show that recent pasta sauce coupon redemptions is positively related to pasta coupon redemptions.
# To do this, we're going to use a simple linear regression with 'coupon_pasta sauce' as our feature (i.e. x) variable and pasta coupon redemptions (coupon_pasta) as our target (i.e. y)
# Let's define these variables...

# In[9]:


X = train['coupon_pasta sauce']
y = train['coupon_pasta']


# It's often useful to plot distributions of feature and target variables using histograms...

# In[10]:


plt.hist(X, bins=100);
plt.xlabel('Pasta Sauce Coupons');
plt.ylabel('Frequency');


# In[11]:


plt.hist(y, bins=100);
plt.xlabel('Pasta Coupons');
plt.ylabel('Frequency');


# ##### Q2: Describe the distributions of each variable. Is there a name for the way that the 'pasta_coupons' variable is distributed? Why do you think it's distributed like this?

# In[12]:


# This cell may be useful for Question 3
X_train = X


# We can use the LinearRegression() function in scikit-learn to run a regression

# In[13]:


mod = linear_model.LinearRegression()
mod.fit(X_train.reshape(-1,1), y);
preds = mod.predict(X_train.reshape(-1,1))


# For regression models, it's often useful to plot predictions against the target. Look out for a regression line with a steep gradient; this is a sign that your regression model has found some signal.

# In[14]:


plt.scatter(X, y) # Creates a scatter plot of the X against the Y
plt.plot(X, preds, '.k',
         linewidth=3); # Plot a black line showing the X against the predictions
plt.xlabel('Pasta Sauce Coupons')


# ## 2. Evaluating regression model fit

# After fitting a model, it's crucial to evaluate how well it has fit the data.
# 
# In particular, there are two key metrics that are useful for evaluating the fit of a regression model:
# 
# 1. R-squared (higher=better fit)
# 2. RMSE (higher=worse fit)
# 
# Thankfully, sklearn have created a lot of handy functions for evaluating model fits, including the above metrics

# In[15]:


# First, let's evaluate the R-squared of the predictions
from sklearn.metrics import r2_score
r_squared = r2_score(y, preds)
print('Our regression model explains {0:.1f}% of the variance in our data' .format(r_squared * 100))
# Documentation for this metric is here: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html


# In[16]:


# Now let's calculate the root-mean-squared-error (RMSE) of our regression model
# The RMSE is useful for comparing two regression models. If we don't have a competitor model, it can be useful to compare it against a vector of ones and a vector of zeros
from sklearn.metrics import mean_squared_error
RMSE = np.sqrt(mean_squared_error(y, preds))
print ('The RMSE of a dumb model that predicts 1s only is {0:.3f}' .format(np.sqrt(mean_squared_error(y, np.ones(len(preds))))))
print ('The RMSE of a dumb model that predicts 0s only is {0:.3f}' .format(np.sqrt(mean_squared_error(y, np.zeros(len(preds))))))
print ('The RMSE of a dumb model that predicts the mean only is {0:.3f}' .format(np.sqrt(mean_squared_error(y, np.repeat(np.mean(y), len(preds))))))
print ('The RMSE of our regression model is {0:.3f}' .format(RMSE))


# ##### Q3: What does the black trend line (i.e. regression line) and the model performance tell you about the relationship between pasta sauce coupon redemptions and pasta coupon redemptions? What would you tell your client at Ocset about their hypothesis (i.e. that pasta sauce coupon redemptions are predictive of pasta coupon redemptions)?
# 

# The model isn't the best, there is a relationship between the feature and the predict variable as the RMSE of the model is < than the model that predicts the meant.
# The line of the prediction show some slope, if it was useless totally it would be flat.
# It isn't a great model, but it's better than using the mean.

# ## 3. Visualising Gradient Descent
# 
# Gradient descent is often used to minimise a regression function and is used in many modern machine learning algorithms (e.g. Neural Networks).
# 
# 
# The purpose of gradient descent is to minimise a function. In this case, our function is:
# 
# $$RSS = J(\theta_0,\theta_1) = \sum_{t=1}^{m}\frac{1}{2}(y - f(x)))^2$$
# 
# Where:
#     
# $$f(x) = \theta_0 + \theta_1 x$$
# 
# Recall that $\theta_0$ determines the intercept of the line whilst $\theta_1$ determines the slope. Gradient descent tries to learn the optimal values for these parameters so as to minimise our objective/cost function.

# In[17]:


X = train['coupon_pasta sauce']
y = train['coupon_pasta']


# In[18]:


# First, let's define a gradient descent function (don't worry too much if you dont understand it)
def gradient_descent(x, y, iters, alpha):
    """Gradient descent function written by http://tillbergmann.com/blog/python-gradient-descent.html"""
    costs = []
    m = y.size # number of data points
    theta = np.random.rand(2) # random start
    history = [theta] # to store all thetas
    preds = []
    for i in range(iters):
        pred = np.dot(x, theta)
        error = pred - y 
        cost = np.sum(0.5 * (error)) ** 2
        costs.append(cost)
        
        preds.append(np.dot(x, theta))

        gradient = x.T.dot(error) / m
        theta = theta - alpha * gradient  # update
        history.append(theta)
        
    return history, costs, preds


# In[19]:


# We need to add an additional vector of ones to our X, which is used for the intercept (theta_0)
X = np.c_[np.ones(X.shape[0]), X] 


# We've defined our gradient descent function and set up our X matrix. Now we need to run gradient descent.
# 
# The alpha variable determines the rate at which the algorithm learns. When an alpha is higher, gradient descent will take larger steps across the error surface.
# 
# The iters variable determines the number of iterations that gradient descent will make. 

# In[20]:


alpha = 0.01 # set step-size
iters = 300 # set number of iterations

history, cost, preds = gradient_descent(X, y, iters, alpha)
theta = history[-1]


# If gradient descent is converging, then the error/cost will get lower over time. Let's plot that.

# In[55]:


# Plot the learning curve, with iterations on the X axis and cost on the y axis
x1, y1 = (zip(*enumerate(cost)))
plt.plot(x1,y1);


# ##### Q4: Try changing the alpha value. What happens to the learning curve when you a) increase the apha b) decrease the alpha. Now try changing the number of iterations. How many iterations are required to reach a minimum in the cost? How does this change as you vary the alpha value?
# 
# (Note that - in practice - you would not have to vary alpha or the number of iterations when fitting a regression model. The defaults are sufficient on 99.9% of occasions. This is just to illustrate what is happening in the background.)

# As before, we can plot the predictions from our regression line. We can actually do this for every iteration made during gradient descent.
# 
# To get a sense of how gradient descent learns, try running the code below for different values of 'iteration'. Recall that the first iteration of gradient descent  (iteration = 0)  is a random guess, so it's unlikely that this line will be very good. With more iterations (as iteration gets larger), the line should look more like a fitted regression line. Eventually the line will stop moving. Can you see how this links up with the learning curve above?

# In[21]:


iteration = 1
plt.scatter(X[:,1], y);
plt.plot(X[:,1], preds[iteration], '.k',
         linewidth=3);


# ## 3.2 BONUS: Contour plots

# Another neat way to understand gradient descent is to think about how it moves around the error surface.
# 
# The error surface can be thought of as a bowl. The bottom of the bowl is the 'global minimum'. It is gradient descent's job to find the bottom of the bowl. It does this by using the gradients of the nearby surface to move in a downward direction.
# 
# Contour plots help us to visualise the error surface in two dimensions. Concentric rings that are small, blue and close together indicate the bottom of the error bowl, whilst larger, red rings that are more widely spaced apart indicate the top (open part) of the bowl.

# In[58]:


# First, let's define a neat function that calculates the cost for a given set of thetas
# Code adapted from: https://gist.github.com/marcelcaraciolo/1321575
def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
    m = y.size

    predictions = X.dot(theta).flatten()

    sqErrors = (predictions - y) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J


# In[59]:


# Define a grid of theta values that we want to calculate the cost J for
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# Initialize J_vals to a matrix of 0's
J_vals = np.zeros(shape=(theta0_vals.size, theta1_vals.size))

# Compute the cost for each set of theta values
for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = np.zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        J_vals[t1, t2] = compute_cost(X, y.values, thetaT)


# In[60]:


# Using Pylab, generate a contour plot
contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(0.01, 1.6, 30), cmap=plt.get_cmap('RdBu_r'))
plt.ylabel('theta_1')
# Let's also plot the 'optimal' thetas found by gradient descent; this should be in the bottom of the bowl (inside the blue rings)
plt.scatter(theta[0], theta[1])
plt.show()

