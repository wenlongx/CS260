# work on code until like 8pm, start putting together slides
# run and generate some images


DAE and Stacked DAE
    - train at various noise levels
    - trained on MNIST images (not adversarial)

MNIST CNN -> train it on just normal images
    see how much normal error rate is on mnist cnn
generate adversarial examples on MNIST CNN
    see how much the adv error rate is against MNIST cnn

train DAE on the mnist dataset for several noise levels
use DAE to preprocess adversarial examples and also normal examples
    see how much normal error rate is on preprocessed
    see how much adv error rate is on preprocessed

generate images for images passed through DAE
generate images for adv passed through DAE

compare DAE against another method (what method?)
    fgsm is a white box attack
    look for some white box defense (PGD adv training?)
    perturb examples at same rate, use defense method in mnist
    - adversarial training?
