import numpy as np

```
Forward Pass

Z1 = X.W1 + b1
A1 = ReLU(Z1)  
Z2 = A1.W2 + b2
exp_scores = exp(Z2)  
probs = exp_scores / sum(exp_scores)


Backward Pass

delta3 = probs
delta3[range(len(X)), y] -= 1
dW2 = A1.T.dot(delta3)
db2 = sum(delta3)
delta2 = delta3.dot(W2.T) * (A1 > 0)
dW1 = X.T.dot(delta2)
db1 = sum(delta2)
```


