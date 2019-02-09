# Importance Sampling
Important sampling: Current SGD algorithm selects each batch with uni- form sampling. Intuitively this is not the best way to sample—there should be some “easy” samples that already well-trained and doesn’t need more updates; while some “hard” samples require more frequent updates. The question is how to estimate the “importance” of each sample, and how to do it efficiently? Several references [19, 20]. You can try to implement these methods and investigate their weakness, and try to come up with better approaches.


# Adversarial Networks generate examples (11, 12)
Attack: How to generate adversarial examples for a given model? In the white-box setting (the attacker knows the model structure & weights) and for standard image classification this is considered a solved problem [10], but how to do this in a different task (e.g., object detection, video classifi- cation) is still interesting. Also, it is interesting to study how to generate adversarial examples in the probability black box setting (when you can observe the probability of model’s output), see some recent work [11, 12]. A more challenging problem is the black-box hard label setting where you can only query the model and get the hard label output, see recent work [13, 14]. Black-box attacks are still open problem since they typically require large number of queries. A comparison between current methods or improvement will be a good project.

# Adversarial Networks Defense (15, 16)
Defense: Defense (improving robustness with respect to adversarial at- tacks) is still an open problem. Two viable approaches are adversarial training [15] and randomization [16]. A comparison between current defense algorithms or improvement will be a good project.
