# adversarial
## general adversarial
### Explaining and Harnessing Adversarial Examples
- intro
    - several ML models are vulnerable to adversarial examples (misclassify examples only slightly different from correctly classified examples drawn from the same distribution)
    - adversarial == fundamental blindspots of training
    - what causes them?
        - extreme nonlinearity of networks?
        - insufficient model averaging?
        - insufficient regularization?
    - no:
        - linear behavior in high dimensional spaces is sufficient to cause adv. examples
        - exploit this to generate examples quickly
    - adv training / examples can provide regularization benefit besides just dropout alone
    - generic regularization doesn't help adv., but changing to nonlinear model families like RBF helps
    - **this suggests that models are easy to train cuz of linearity, but nonlinear effects resist adversarial perturbation**

- related work
    - modern ML aren't learning true underlying concepts
    - doesn't work on points from distr. that have low prob of occurring
    - bad bc modern approach to CV is to use euclidian distance, but that doesn't necessarily mean same classes if they're close
    - $w^Tx = w^Tx + w^Tn$
        - adversarial perturbation causes activation to grow by some amount
    - in a linear example, many small changes can cause large change in activation
    - this is their hypothesis
- linear perturbation of non-linear models
    - NN are too linear to resist linear adversarial perturbation
    - LSTM / ReLU / Maxout networks are designed to behave very linearly so that they are easier to optimize (nonlinear models, like sigmoid spend most of time in linear regime)
    - fast gradient sign method
        - take gradient of the loss, find sign of that, then multiply by small delta
        - needs to know the loss function and gradients though - not a black box
- adversarial training of linear models vs weight decay
    - add a penalty based on avoiding misclassification of perturbation
- adversarial training on deep NN
    - deep networks are able to represent functions that resist adversarial perturbation
    - added a penalty for Adversarial examples, and became a little bit resistant
    - can be seen as minimizing worst case error when data is perturbed by an adversary
    - minimizing an upper bound on expected cost over noisy samples with noise from Uniform added to inputs
    - form of active learning, but select labels from nearby points
- why do adversarial examples generalize?
    - examples are abundant - form the rest of the space not taken up by one class or the other
- summary
    - adversarial examples = property of high-dimensional dot products
    - result of models being too linear
    - different models learn similar functions for same task
        - adversarial examples are highly aligned with weight vectors
    - direction of perturbation rather than point matters most
    - adversarial training == a form of regularization


### Explaining and Harnessing Adversarial Examples
