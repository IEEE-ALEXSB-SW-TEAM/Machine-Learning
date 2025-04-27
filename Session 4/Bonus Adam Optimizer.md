---
title: 'Bonus: Adam Optimizer'

---



## 🧠 What is the Adam Optimizer?

Adam stands for:  
**Adaptive Moment Estimation**  

It's one of the most popular **optimization algorithms** used in training deep learning models.  
Adam is like a smart and upgraded version of **Stochastic Gradient Descent (SGD)**.  
It **adjusts the learning rate** for each parameter **automatically** and **uses past gradients** to make training faster and more stable.

---

## 🧩 How Does It Work?

Adam keeps track of two things during training:

1. **The average of gradients** → called the **first moment** (like momentum).
2. **The average of squared gradients** → called the **second moment** (like the variance).

> These are used to make better updates to weights.

---
## 🔢 Let’s Start with “Moments” in Math

In statistics, the **moment** of a distribution is a way to describe its **shape**.

| Moment Type     | Formula (for a variable x)         | Meaning                         |
|------------------|-------------------------------------|----------------------------------|
| **1st moment**   | `E[x]`                              | Mean (average)                  |
| **2nd moment**   | `E[x²]`                             | Raw variance (if centered: true variance) |
| **k-th moment**  | `E[x^k]`                            | Higher order shape descriptors |

So, in Adam:
- The **1st moment** of the gradients → `E[gₜ]` → gives you an average of the gradients → this is called **momentum** in optimization.
- The **2nd moment** of the gradients → `E[gₜ²]` → gives you an average of the *squares* of gradients → which relates to **variance**.

So they are using terminology from **statistics** and not just optimization.

---

## 🔁 Now... Why Not Just Say "Momentum and Variance"?

Because:
- Adam is based on **statistical moments**, not exactly the same as classical "momentum" in physics.
- In classical **Momentum optimization**, you don’t use bias correction or square the gradient.
- In classical **variance**, you subtract the mean — Adam doesn’t do that either (it uses raw second moment).

> So to keep it general and precise, researchers use the term:
> - "1st moment" instead of “momentum”  
> - "2nd moment" instead of “variance”

It’s a way to say:  
🧠 “This is *like* momentum and *like* variance — but not exactly.”

---

## 🧮 Look at the Equations to Understand the Link

### 🥇 First Moment (Mean of gradients → like momentum):

$mₜ = β₁ * mₜ₋₁ + (1 - β₁) * gₜ$

This is an **Exponential Moving Average (EMA)** of gradients → it's smoothing the current gradient by mixing it with the past gradients.

This is exactly what **momentum** in optimization does — it remembers the direction you’ve been heading and builds inertia.

✅ Why call it momentum?
- It “carries the gradient forward”.
- Helps accelerate training in consistent directions.

---

### 🥈 Second Moment (Mean of squared gradients → like variance):

$vₜ = β₂ * vₜ₋₁ + (1 - β₂) * gₜ²$

This is the **EMA of the squared gradients**.

It captures how **large** or **bumpy** the gradients are — without considering direction (since it’s squared).

This acts like a **proxy for variance**, because:
- Large gradient → big $vₜ$ → maybe unstable → slow down updates.
- Small gradient → small $vₜ$ → stable → take normal-sized step.

✅ Why call it variance?
- It tells you the **spread** or **volatility** of the gradients.
- Used to **scale the update** inversely — large variance → smaller step.


---

## 🔢 The Formula and Terms

Let’s go step-by-step:

- Let’s say:
  - $gₜ$ = current gradient at time step $t$
  - $mₜ$ = moving average of gradients (1st moment)
  - $vₜ$ = moving average of squared gradients (2nd moment)
  - $θₜ$ = parameters (weights) at time step $t$
  - $α`$= learning rate (e.g., 0.001)
  - $β₁$ = decay rate for the 1st moment (commonly 0.9)
  - $β₂$ = decay rate for the 2nd moment (commonly 0.999)
  - $ε`$= small value to avoid division by zero (e.g., 1e-8)

---

## 🧮 Step-by-Step Calculation

### 1. **Initialize**
$m₀ = 0$
$v₀ = 0$
$t = 0$


### 2. **At every time step t**:
Increment time:  
$t = t + 1$

#### ➤ Update the **1st moment (mₜ):**
$mₜ = β₁ * mₜ₋₁ + (1 - β₁) * gₜ$
👉 This is the exponential moving average of gradients (like momentum).  
👉 **β₁ is the decay rate** — controls how much we remember past gradients.

#### ➤ Update the **2nd moment (vₜ):**
$vₜ = β₂ * vₜ₋₁ + (1 - β₂) * (gₜ)^2$
👉 This is the exponential moving average of squared gradients.  
👉 **β₂ is the decay rate** — controls how much we remember past squared gradients.

> A **decay rate** (like β₁=0.9) means:
> - Keep 90% of the previous value.
> - Add 10% of the new value.
> So the past has **more influence** but we slowly mix in new info.

#### ➤ **Bias Correction** (important for early steps):
$m̂ₜ = mₜ / (1 - β₁ᵗ)$
$v̂ₜ = vₜ / (1 - β₂ᵗ)$
👉 These fix the fact that $mₜ$ and $vₜ$ start at 0 and are biased early on.

#### ➤ **Update Parameters (weights):**
$θₜ = θₜ₋₁ - α * m̂ₜ / (√v̂ₜ + ε)$

This is the actual update.  
- If $v̂ₜ$ is large (i.e. the gradient is very bumpy), the step is **smaller**.  
- If $m̂ₜ$ is large and consistent, the step is **bigger**.  
- This is why Adam adapts smartly.

---

## 🔍 Summary of Hyperparameters

| Symbol  | Meaning                        | Typical Value |
|---------|--------------------------------|---------------|
| α       | Learning rate                  | 0.001         |
| β₁      | Decay rate for 1st moment      | 0.9           |
| β₂      | Decay rate for 2nd moment      | 0.999         |
| ε       | Numerical stability constant   | 1e-8          |

---

## ✅ Why Use Adam?

- **Faster convergence** – learns faster than plain SGD.
- **Stable** – works well even with noisy gradients.
- **Adaptive** – handles different parameter types automatically.

---

## 🧠 Real-Life Analogy

Imagine you're hiking down a mountain (minimizing the loss function):

- **Gradient** tells you the slope at each step.
- **Momentum (m)** remembers the direction of your past steps → helps you speed up.
- **Variance (v)** tells you how bumpy or smooth the path is → helps you slow down in shaky areas.
- **Adam** combines both — it remembers your past and adjusts your step size accordingly.

---



