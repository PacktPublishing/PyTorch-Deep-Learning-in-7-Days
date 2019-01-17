#%%
# understanding gradients is a bit of math, but we'll try to keep this
# simple -- bascially a numerical gradient tells you which direction
# you need to move when you are machine learning -- positive or
# negative -- and well as having an actual numerical value
# that tells you 'how much' you should move

# a machine learning loop takes the gradeients for a group
# of tensor operations, and then updates the value of the tensors
# that are being 'learned' using the product of the gradients
# and the learning rate

# so if you know a bit of math, you know a gradient is a numerical
# representation of a derivative -- which means you know to ask
# 'a gradient with respect to what?' -- and the answer there
# is a loss function, you use gradients to figure the direction
# to go to make your loss smaller

# here is the simplest example from grade school algebra I could
# cook up -- before you knew algebra -- this was a bit of a 
# mystery how to solve it, and I know I personally tried -- 
# just plain guessing the numbers to 'solve' equations --
# machine learning is a bit like that, but instead of just plain
# guessing, we use the gradient to figure how far off, and what
# our next guess should be --= OK

#%%
import torch

# X + 1 = 3 -- that's our little bit of algebra

# here is our random initial guess
X = torch.rand(1, requires_grad=True)
# and our formula
Y = X + 1.0
Y

#%%
# now, our loss is -- how far are we off from 3?
def mse(Y):
    diff = 3.0 - Y
    return (diff * diff).sum() / 2

#%%
# the gradient on our X -- that tells us which direction
# we are 'off' from the right answer -- let's look when we are too high
loss = mse(Y)
loss.backward()
X.grad

#%%
# now -- let's use that gradient to solve some grade school
# algebra with simple machine learning
learning_rate = 1e-3
# here is our learning loop
for i in range(0, 10000):
    Y = X + 1.0
    loss = mse(Y)
    # here is the 'backpropagation' of the gradient
    loss.backward()
    # and here is the 'learning', so we turn off the graidents
    # from being updated temporarily
    with torch.no_grad():
        # the gradient tells you which direction you are off
        # so you go in the opposite direction to correct the problem
        X -= learning_rate * X.grad
        # and we zero out the gradients to get fresh values on 
        # each learning loop iteration
        X.grad.zero_()
# and -- here is our answer
X
# OK -- you can see that this is approximate -- and that's an
# important point -- machine learning is going to approximate
# and you can control how close you get to the target answer
# by altering your learning rate or your number of iterations
# experiment with this by altering the `learning_rate`
# and the number of loops in `range`