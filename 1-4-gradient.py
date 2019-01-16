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

import torch

# X + 1 = 3 -- that's our little bit of algebra

# here is our random initial guess
X = torch.rand(1, requires_grad=True)
# and our formula
yhat = X + 1.0
yhat

#%%
# now, our loss is -- how far are we off from 3?
loss = 3.0 - yhat
loss

#%%
# now -- let's use that gradient to solve some grade school
# algebra
learning_rate = 1e-3
# here is our learning loop
for i in range(0, 10000):
    yhat = X + torch.tensor([1.0])
    loss = torch.tensor([3.0]) - yhat
    # here is the 'backpropagation' of the gradient
    loss.backward()
    # and here is the 'learning', so we turn off the graidents
    # from being updated temporarily
    with torch.no_grad():
        X -= learning_rate * X.grad
        # and if we have no gradient left -- it has vanished
        # we are done
        if X.grad.sum() == 0:
            print('gradient vanished', X.grad)
            break
        if loss.sum() < learning_rate:
            print('no loss', loss)
            break
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