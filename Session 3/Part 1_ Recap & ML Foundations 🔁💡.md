---
title: "Part 1: Recap & ML Foundations \U0001F501\U0001F4A1"

---

# Part 1: Recap & ML Foundations 🔁💡


## 📚 Regression vs Classification

| Task Type      | Output              | Example                              | Models Used                    |
|----------------|---------------------|--------------------------------------|--------------------------------|
| Regression     | Continuous number   | Predict house price                  | Linear Regression              |
| Classification | Categorical label   | Predict if email is spam or not      | Logistic Regression, KNN, Trees|

---

## 🧠 Model Recap

### 📈 Linear Regression

Used for regression problems. It tries to fit a straight line to data:

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$

Goal: minimize Mean Squared Error (MSE)

---

### 📉 Logistic Regression

Used for classification. It outputs **probability** of a class using a sigmoid:

$$
P(y=1 \mid x) = \frac{1}{1 + e^{-(w^T x)}}
$$

- Output between 0 and 1
- Threshold at 0.5 to decide the class
- Trained using cross-entropy loss
---
# Softmax in Logistic Regression

## Introduction

In Logistic Regression, when we have more than two classes (i.e., a multi-class classification problem), we need to extend the basic binary logistic regression model. The **Softmax function** is commonly used in such cases, especially in **multinomial logistic regression**.

The **Softmax function** takes a vector of raw scores (often called **logits**) and converts them into probabilities by normalizing the output values. Each probability corresponds to the likelihood of a given input belonging to a specific class.

## Formula

Given a vector of raw scores (logits) $z = [z_1, z_2, ..., z_K]$, the Softmax function computes the probability of class $( i )$ as follows:

$$
P(y = i | x) = \frac{e^{z_i}}{\sum_{k=1}^{K} e^{z_k}}
$$

Where:
- $P(y = i | x) )$ is the probability that input $( x )$ belongs to class $( i )$.
- $( z_i )$ is the raw score (logit) for class $( i )$.
- $( K )$ is the total number of classes.
- The denominator $( \sum_{k=1}^{K} e^{z_k} )$ is the sum of the exponentials of all logits, ensuring that the sum of all class probabilities equals 1.

## How it Works

1. **Raw Scores (Logits)**: In a classification problem, we typically have raw outputs (logits) from a model, such as a neural network, that correspond to each class.
   
2. **Exponentiation**: Softmax exponentiates each of these raw scores, making them positive and larger for higher logits.

3. **Normalization**: The sum of all these exponentiated scores is computed and used to normalize the individual values. This step ensures that the probabilities add up to 1, which is a requirement for probability distributions.

4. **Interpretation**: After applying Softmax, the output for each class represents the probability that the input \( x \) belongs to that class. The class with the highest probability is chosen as the model’s prediction.

## Use Case in Logistic Regression

In binary logistic regression, we model the probability of one class (e.g., class 1) using the sigmoid function. In multinomial logistic regression (for multi-class classification), we use the Softmax function instead. Each class gets a probability score, and we can select the class with the highest probability as the final prediction.

### Example:

Suppose we have a three-class problem, with raw scores (logits) for a given input \( x \) as follows:

$$
z = [2.0, 1.0, 0.1]
$$

To compute the probabilities, we apply the Softmax function:

$$
P(y = 1 | x) = \frac{e^{2.0}}{e^{2.0} + e^{1.0} + e^{0.1}} \approx 0.659
$$
$$
P(y = 2 | x) = \frac{e^{1.0}}{e^{2.0} + e^{1.0} + e^{0.1}} \approx 0.242
$$
$$
P(y = 3 | x) = \frac{e^{0.1}}{e^{2.0} + e^{1.0} + e^{0.1}} \approx 0.099
$$

Thus, the model predicts that the input belongs to class 1, as it has the highest probability (0.659).

## Conclusion

Softmax is a powerful tool for multi-class classification, providing a way to model probabilities for each class. It ensures that the model's output can be interpreted as a probability distribution over the classes, making it useful in tasks like image classification, text categorization, and more.

---

### 📍 K-Nearest Neighbors (KNN)

- **Lazy learner** (no training phase)
- To predict a label: look at the labels of the K closest points
- Distance metric: usually Euclidean

#### Pros:
- Simple, effective
- No training time

#### Cons:
- Slow with large datasets
- Sensitive to irrelevant features

---

## 🧪 A Typical ML Pipeline

1. **Understand the problem**  
   → Classification or regression?

2. **Collect & clean data**  
   → Handle missing values, duplicates

3. **Preprocessing**  
   - Normalize/standardize data
   - Encode categorical variables
   - Feature selection

4. **Split data**  
   → Train / Test (e.g., 80/20 split)

5. **Train a model**  
   → Use scikit-learn, PyTorch, etc.
   Note: for learning purposes, you can try make the model yourself.

6. **Evaluate performance**
   - Accuracy (classification)
   - MSE/R² (regression)

7. **Optimize model**  
   - Tune hyperparameters
   - Try better features or different models

8. **Deploy**  
   → Build an app, API, or dashboard

---

# Evaluation Metrics 🧪🎯

When you're building a classification model, accuracy is **not always enough** — especially with **imbalanced datasets** (e.g., 95% negative, 5% positive).

That's why we use more detailed metrics:  

**Precision, Recall, and F1 Score** 🔍

---

## 📦 Confusion Matrix

A table that summarizes how well a classifier performs:

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP)     | False Negative (FN)    |
| **Actual Negative** | False Positive (FP)    | True Negative (TN)     |

---

## 🎯 Accuracy

The simplest metric:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

> **When is it useful?**  
Useful when **classes are balanced** and **all errors are equally bad**.

> **Example**  
In digit recognition (MNIST), where each class is balanced, accuracy is a good measure.

---

## 📌 Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

How many of the **positive predictions** were **actually correct**?

> **When is it important?**  
When **false positives** are costly.

> **Example**  
-> **Email Spam Detection**:  
  Precision matters — if the model wrongly marks a real email as spam (false positive), the user might miss something important.
-> **Cheating Detection**:  
  Precision matters — if the model wrongly marks a student as cheater (false positive), the student will be wronged.

---

## 🔍 Recall (Sensitivity / True Positive Rate)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

How many of the **actual positives** did the model **correctly find**?

> **When is it important**
> When **false negatives** are costly.

> **Example**  
-> **Cancer Detection**:  
  Recall matters — we don’t want to miss a real cancer case (false negative), even if it means getting some false alarms.

---

## ⚖️ F1 Score

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

A **harmonic mean** of precision and recall. Balances both.

> **When is it useful?**  
When you want to balance precision and recall, especially on **imbalanced datasets**.

> **Example**  
-> **Fraud Detection**:  
  You care about both catching fraud (recall) and minimizing false alarms (precision), so F1 is perfect.

---
## 🧪 Real-World Summary

| Task                      | Priority Metric | Why? |
|---------------------------|------------------|------|
| Email Spam Filter         | Precision        | Don't mark important emails as spam |
| Cancer Diagnosis          | Recall           | Don’t miss a real cancer case |
| Face Recognition Login    | Precision        | Don’t allow unauthorized access |
| Fraud Detection           | F1 Score         | Balance catching fraud & avoiding false alarms |
| Balanced Dataset (e.g. MNIST) | Accuracy       | All classes matter equally |


✅ Now that we’ve made our recap, let’s dive into **Decision Trees** — one of the most interpretable ML models.
