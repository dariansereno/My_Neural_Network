# DeepLearningKit

**Description:**

Welcome to DeepLearningKit, a Python library crafted with a passion for learning and a drive to simplify the journey into the depths of neural networks. As my inaugural project in this domain, DeepLearningKit is designed primarily for my educational purposes, aiming to help me to understand and experiment with deep learning concepts.
This release marks the first version of DeepLearningKit, but the journey doesn't end here! I'll be actively working on improving and expanding the library regularly. Stay tuned for updates and new features in future releases.

**Getting Started:**
1. **Installation:**
   ```
   pip install deeplearningkit
   ```

2. **Usage:**
   ```python
    import deeplearningkit as nn
    import numpy as np

 	# Load data
    X_train, y_train = load_data()

    # Define and compile the model
    model = nn.Model()
	model.add(nn.Layer.Dense(train_X.shape[1], 24,  nn.initializer.He(train_X.shape[1])), nn.Activation.ReLU())
	model.add(nn.Layer.Dense(24, 24,  nn.initializer.He(24)), nn.Activation.ReLU())
	model.add(nn.Layer.Dense(24, 4,  nn.initializer.He(24)), nn.Activation.Softmax_CategoricalCrossEntropy())
	model.compile(nn.Optimizer.Adam(), nn.Loss.CategoricalCrossEntropy())

	# Train the model
	model.fit(x=X_train, y=y_train, batch_size=32, epochs=1001, shuffle=True, display=True, plot=True)

	# Evaluate the model
	X_test, y_test = load_test_data()
	model.evaluate(X_test, y_test)
   ```

**License:**
DeepLearningKit is licensed under the [MIT License](LICENCE).

**Acknowledgments:**
This project wouldn't have been possible without the support and guidance from the deep learning community, bigup to the book : 'Neural Network From Scratch' by Harrison Kinsley and Daniel Kukiela <3 . Let's continue to learn!
