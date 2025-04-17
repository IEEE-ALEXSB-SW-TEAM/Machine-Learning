---
title: "Part 2: Decision Trees \U0001F333"

---

# Part 2: Decision Trees 🌳

## 🌱 What is a Decision Tree?

A **Decision Tree** is a supervised machine learning algorithm that makes decisions based on answering a series of questions about the input features.

Think of it as a **flowchart** or a "tree" where:
- **Nodes** represent decisions or questions.
- **Edges** represent the possible outcomes of those decisions.
- **Leaves** represent the final prediction or decision.

---

## 📦 Key Terminology

- **Root Node**: The starting point of the tree, where the first decision is made.
- **Internal Node**: A decision node, where data gets split further.
- **Leaf Node**: A terminal node that holds the final prediction or classification.
- **Depth**: The number of levels from the root node to the leaf nodes.
- **Splitting**: The process of dividing a node into sub-nodes based on a feature.
- **Pruning**: Reducing the size of the tree by removing unnecessary branches (to avoid overfitting).

---

## 🧠 How Decision Trees Work

1. **Choose a feature**: The tree starts by asking a question based on one of the features (e.g., "Is the weather sunny?").
2. **Split the data**: Based on the answer to the question, the data is divided into subsets. For example, if "Is the weather sunny?" is answered as yes, the data with sunny weather goes to one branch.
3. **Repeat**: For each new branch, a further question is asked about one of the remaining features until you reach a leaf node.
4. **Prediction**: When a leaf node is reached, the prediction is made. This could be a class label (for classification) or a continuous value (for regression).

---

## ⚙️ Splitting the Data: How Does the Tree Choose?

The key to a decision tree is how it splits data at each node. The goal is to make each subset as **pure** as possible (i.e., all data points in the subset belong to the same class or are similar).

The most common criteria to decide the best feature for splitting are:

### 1. **Entropy & Information Gain (ID3 Algorithm)**

Entropy is a measure of **impurity** or **uncertainty** in a dataset.

$$
H(S) = -\sum_{i=1}^{n} p_i \log_2(p_i)
$$

Where $p_i$ is the proportion of class $i$ in the dataset. The lower the entropy, the purer the dataset.

- **Information Gain** (IG) measures how much **entropy** is reduced when splitting the dataset on a particular feature:

$$
IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
$$

Where:
- $A$ is the feature we're considering for the split
- $S_v$ is the subset of data where feature $A = v$

The feature that gives the **highest information gain** is chosen to split the data.

### 2. **Gini Impurity (CART Algorithm)**

Another measure of impurity, used in the **CART** (Classification and Regression Trees) algorithm.

$$
Gini = 1 - \sum_{i=1}^{n} p_i^2
$$

Where $p_i$ is the proportion of class $i$ in the node.

The feature with the **lowest Gini impurity** is chosen for the split.

---

## 🛠 Decision Tree Algorithms

- **ID3**: Uses **Information Gain** to decide the best feature for splitting.
- **C4.5**: A more advanced version of ID3 that also uses **Gain Ratio** (to avoid bias towards features with more levels).
- **CART**: The most commonly used tree algorithm (especially in scikit-learn). It can be used for both classification (with Gini impurity) and regression (using Mean Squared Error).

---

## 🌳 Decision Tree Example

Imagine we have a [simple dataset](https://www.cs.cmu.edu/~aarti/Class/10315_Fall20/recs/DecisionTreesBoostingExampleProblem.pdf) for predicting whether someone will play tennis based on weather conditions:

![Capture](https://hackmd.io/_uploads/BJOG2HnRke.png)



Let's now manually implement a decision tree using entropy and Gini index as the criteria, and then test it on our notebook.


---
## 🌱 Splitting on Continuous Values

When dealing with continuous features, the decision tree must choose a **threshold** for splitting the data at each node. For example, instead of asking "Is the weather sunny?", the question might be "Is the temperature greater than 75 degrees?"

Here’s how it works:
1. **Choose the feature**: For each feature, the algorithm considers all possible thresholds (like temperature > 75, temperature > 70, etc.).
2. **Split the data**: The dataset is divided into two subsets — one where the feature is greater than the threshold, and one where it is less.
3. **Calculate impurity**: The algorithm calculates the **impurity** (using metrics like **Gini** or **Information Gain**) for each threshold and picks the one that best separates the data.

---

### 📊 Example: Splitting on Continuous Values

What if our dataset has continuous values for **Temperature**


Imagine we want to split on **Temperature**.

The algorithm will consider thresholds like:
- **Temperature > 15**
- **Temperature > 10**
- **Temperature > 25**

Let's calculate the **Information Gain** for each possible threshold. 

After calculating the information gain for each possible threshold, the algorithm will choose the threshold that maximizes the information gain.

You can also try working with the median (time-saving but less accuracy).

---

## 🧩 Pros & Cons of Decision Trees

### ✅ Pros:
- **Interpretable**: You can visualize the tree and explain how decisions are made.
- **Handles Mixed Data**: Can handle both numerical and categorical features.
- **No Feature Scaling**: Doesn't require scaling (e.g., normalization).

### ❌ Cons:
- **Overfitting**: Decision trees tend to overfit, especially with deep trees.
- **Instability**: Small changes in data can result in a completely different tree.
- **Biased**: They can be biased toward features with more categories.

---

## ✂️ Preventing Overfitting

To make sure your decision tree doesn’t overfit, you can use these techniques:

### 1. **Pre-Pruning** (Stop Early)
- Limit tree depth
- Set a minimum number of samples required to split a node
- Set a threshold for node purity (Gini < 0.1)

### 2. **Post-Pruning** (After Tree is Built)
- Cut back branches that provide little to no improvement in accuracy
- Use **cost-complexity pruning** (CART)

### 3. **Ensemble Methods**
- **Random Forest**: Combines multiple decision trees to reduce overfitting and improve accuracy.
- **Gradient Boosting**: Builds decision trees sequentially, where each tree tries to fix the errors made by the previous tree.

---
# Ensemble Learning, Random Forest, and AdaBoost 🌳🤖

## Ensemble Learning 🤝

**Ensemble learning** is a technique in machine learning where multiple models (often weak learners) are combined to make a stronger, more accurate model. The key idea behind ensemble methods is to **combine multiple models** to improve performance and generalization.

There are two main types of ensemble methods:
1. **Bagging (Bootstrap Aggregating)** 🍂
2. **Boosting** ⚡

---

### 1. Bagging (Bootstrap Aggregating) 🍂

**Bagging** focuses on reducing the variance of the model by training multiple models independently, each on a random subset of the data, and combining their predictions.

- **Key Characteristics**:
  - **Parallel Models**: Models are trained independently.
  - Each model is trained on a **bootstrap sample** (random samples with replacement).
  - The final prediction is obtained by **averaging** (for regression) or **voting** (for classification) from all models.

- **Advantages**:
  - **Reduces overfitting** by averaging models.
  - Works well with **unstable models** (like decision trees).

- **Popular Algorithm**:
  - **Random Forest** 🌲.

---

### 2. Boosting ⚡

**Boosting** is an ensemble method where models are trained **sequentially**. Each model corrects the errors of the previous one.

- **Key Characteristics**:
  - Models are trained **sequentially**.
  - Each new model focuses more on **misclassified** data from the previous model.
  - The final prediction is a **weighted combination** of all models.

- **Advantages**:
  - Boosting **dramatically increases accuracy**.
  - It focuses on difficult-to-predict data points.

- **Popular Algorithms**:
  - **AdaBoost** ✨
  - **Gradient Boosting** (e.g., XGBoost, LightGBM) 🚀.

---

## Random Forest 🌳

**Random Forest** is one of the most popular ensemble algorithms and belongs to the **bagging** family. It improves upon decision trees by reducing their variance and helping prevent overfitting.

### What is Random Forest?

Random Forest is an ensemble of decision trees where:
- Each tree is trained on a **random subset of data**.
- Each tree uses a **random subset of features** to make splits.

Once the trees are trained, the final prediction is made by:
- **Classification**: Taking a **majority vote** from all trees.
- **Regression**: Taking the **average** of all trees' predictions.

### How Random Forest Works

1. **Data Sampling**: Each tree is trained on a **bootstrap sample** (random sample with replacement) from the training data.
2. **Feature Selection**: At each node in the tree, a **random subset of features** is selected.
3. **Model Training**: Multiple trees are trained on different data and feature subsets.
4. **Final Prediction**: 
   - **Classification**: Majority vote across all trees.
   - **Regression**: Averaging the outputs of all trees.

### Advantages of Random Forest:
- **Reduces Overfitting**: By averaging over many trees, it prevents overfitting.
- **Handles Missing Data**: Can handle missing values during predictions.
- **Feature Importance**: Helps in understanding the importance of different features.
- **Robust**: Less prone to overfitting compared to a single decision tree.

---

## AdaBoost (Adaptive Boosting) ✨

**AdaBoost** (Adaptive Boosting) is a boosting ensemble method that combines weak learners to create a strong learner. Unlike bagging methods like Random Forest, AdaBoost trains models **sequentially**.

### What is AdaBoost?

AdaBoost works by building multiple models, with each new model trying to **correct the errors** of the previous one. It **adjusts the weights** of misclassified examples, giving them more importance in the next iteration.

### How AdaBoost Works

1. **Initialize Weights**: Each training example starts with an equal weight.
2. **Train a Weak Learner**: A simple model (e.g., decision stump) is trained on the data.
3. **Update Weights**: After each model is trained, AdaBoost **increases the weight** of misclassified examples, making them more important for the next model.
4. **Repeat**: This process is repeated until the desired number of models are trained or until no further improvement can be made.
5. **Final Prediction**: The final prediction is a **weighted sum** of predictions from all models.

### Advantages of AdaBoost:
- **Improves Accuracy**: AdaBoost can turn weak learners into strong models by correcting errors.
- **No Overfitting**: Focuses on fixing mistakes, reducing overfitting.
- **Efficient**: Works well even with simple models like decision stumps.

### Disadvantages of AdaBoost:
- **Sensitive to Noisy Data**: AdaBoost can overfit if there is a lot of noise or mislabeled data.
- **Training Time**: Since models are trained sequentially, it can be slower compared to parallel methods like Random Forest.

---

## Key Differences Between AdaBoost and Random Forest:

| Feature               | **AdaBoost** ✨                                | **Random Forest** 🌳                           |
|-----------------------|-----------------------------------------------|-----------------------------------------------|
| **Model Type**         | Sequential boosting of weak learners          | Parallel bagging of many decision trees       |
| **Overfitting**        | Less prone, but sensitive to noise and outliers| Less prone to overfitting due to averaging    |
| **Training Speed**     | Slower (sequential learning)                  | Faster (parallel learning)                    |
| **Final Prediction**   | Weighted sum of predictions                   | Majority vote (classification) / Average (regression) |
| **Focus**              | Focuses on correcting mistakes from prior models| Reduces variance by averaging over many models|

---

## Conclusion 🏁

- **Ensemble Learning** combines multiple models to improve predictive performance and generalization.
- **Random Forest** is a bagging method that builds multiple decision trees to reduce overfitting and increase accuracy.
- **AdaBoost** is a boosting method that sequentially builds models, focusing on correcting previous errors.

Each of these methods has its own strengths and weaknesses, and the choice between them depends on the specific problem and data characteristics.

---
## 🌟 Summary
Decision trees are powerful models that offer:
- Simple, interpretable decision-making
- A clear structure for classification and regression problems

A few pitfalls (e.g., overfitting) that can be mitigated through pruning and ensemble methods

Now that you understand the theory and algorithms behind Decision Trees, you're ready to dive into **Naive Bayes** for classification in the next part of this session!
