## Baby Neural nets
[https://github.com/thoppe/baby_neural_nets](https://github.com/thoppe/baby_neural_nets)
----------
[TRAVIS HOPPE](http://thoppe.github.io/), [@metasemantic](https://twitter.com/metasemantic)
!(figures/robot_baby.mp4) <<height:480px>> me working on deep learning

====

## Neural Networks and target shapes
Can we watch the network learn?
What does changing the network topology _look_ like?
!(figures/neural_net2.jpeg) <<height:300px>>

!(figures/source_images/AL1/circle.png) <<height:250px>>
!(figures/source_images/AL1/heart.png) <<height:250px>>
!(figures/source_images/AL1/letter_x.png) <<height:250px>>
!(figures/source_images/AL1/target.png) <<height:250px>>

====

## Shallow vs Deep
Multi-Layer Perceptron learning a circle
Brightness equivalent to NN's uncertainty
  
!(figures/circle_shallow-20160807-162737.mp4) <<height:400px>> Shallow, `200`
!(figures/circle_deep-20160807-162702.mp4) <<height:400px>> Deep, `[10]x15`

====

## Activation functions
!(figures/example_relu.png) <<height:250px>>  
!(figures/example_tanh.png) <<height:250px>>

!(figures/deep_heart_relu.mp4) <<height:400px>> ReLU
!(figures/deep_heart_tanh.mp4) <<height:400px>> tanh


=====

## Concavity and Topology

!(figures/letter_x.mp4) <<height:500px>> Letter X (high concavity)
!(figures/bullseye.mp4) <<height:500px>> Bullseye (interesting topology)
  
=====

## Multi color patterns
Only show most common color
!(figures/source_images/london_underground.png) <<height:490px;transparent>> Source image
!(figures/london_underground.mp4) <<height:500px>> Deep MLP `[10]x40`
  
=====

## Multi color patterns
Straight lines difficult for `tanh`
!(figures/source_images/hillary.png) <<height:490px;transparent>> Source image
!(figures/hillary.mp4) <<height:500px>> Deep MLP `[10]x40`
  
=====

## Multi color patterns
Many colors but simple regions
!(figures/source_images/apple.png) <<height:490px;transparent>> Source image
!(figures/apple.mp4) <<height:500px>> Deep MLP `[10]x40`
  
=====

## Details are hard
Pattern doesn't converge after hours of iterations
(needs more complex topology?)
!(figures/source_images/starbucks.png) <<height:490px;transparent>> Source image
!(figures/starbucks.mp4) <<height:500px>> Deep MLP `[10]x40`

=====

## What's next?

Look at CNN (convolutional neural network)

Combine uncertainty information in color plots

"Morph" from one shape to another as network problem shifts


=====
 
    
#  Thanks, you!
Say hello: [@metasemantic](https://twitter.com/metasemantic)
!(figures/chanel.mp4) <<height:500px>>