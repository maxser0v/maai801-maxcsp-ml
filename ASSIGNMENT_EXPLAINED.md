# Understanding the MAAI801 assignment

The assignment asks one main question:

**Can a machine-learning model choose a better variable to change than a simple rule?**

Think of it as fixing a timetable. Several exams clash, and you can move only **one exam**. Which exam should you move to remove the most clashes?

That is the idea behind the whole project.

## 1. The background

A **constraint** is a rule that a solution should follow.

For example, imagine three exams:

- AI
- Databases
- Machine Learning

Each exam needs a time slot. A rule might say: “AI and Databases must happen at different times.”

A **variable** is something we can change—in this example, an exam’s time slot. Its **domain** is the set of allowed choices, such as slots 1, 2, or 3.

A **Constraint Satisfaction Problem**, or **CSP**, tries to find choices that satisfy every rule.

Sometimes satisfying every rule is impossible. A **Max-CSP** tries to satisfy **as many rules as possible**. We measure how good a solution is by counting its broken rules, called **violations**. Fewer violations means a better solution.

## 2. The simple rule we wanted to improve

The supplied method chooses the variable with the **highest percentage of broken rules involving it**.

For example:

| Variable | Rules involving it | Broken rules | Percentage broken |
|---|---:|---:|---:|
| A | 10 | 2 | 20% |
| B | 2 | 2 | 100% |

The rule chooses **B**.

That sounds reasonable, but there is a catch: **being involved in many broken rules does not guarantee that changing the variable will fix them.** The available replacement values might still cause problems.

The assignment asks whether ML can predict the actual benefit more effectively.

## 3. How the research paper connects

The original paper used this percentage idea inside a **Genetic Algorithm**.

A Genetic Algorithm keeps many possible solutions, combines parts of them, makes small changes, and repeatedly searches for better solutions.

Your assignment studies one small part of that process: **choosing which variable to change once**. We did not build the complete Genetic Algorithm.

## 4. What the machine-learning model learns

We used **supervised learning**. This means showing the model examples with known answers.

Each example describes a candidate variable and asks:

> “If we change this variable using the supplied procedure, how many violations will disappear?”

The known answer is called the **target**.

For example:

- Before the change: 6 violations.
- After the change: 3 violations.
- Target improvement: **3**.

The supplied procedure tries the variable’s allowed values and keeps the best improvement. If none helps, it keeps the original value.

## 5. The steps we performed

### Step 1 — Read the assignment and inspect the supplied materials

The instructor already provided the data and code that counts violations and changes variables. We preserved those helpers and built the learning experiment around them.

The data contained:

- **200 problem instances:** different sets of variables and rules.
- **5 starting states per instance:** different initial assignments.
- **8 variables per state:** eight possible variables to consider changing.

That gives **8,000 candidate examples**: 200 × 5 × 8.

### Step 2 — Choose the information the model can use

For each candidate variable, we supplied six measurements:

| Input | Plain meaning |
|---|---|
| Variable degree | How many rules involve this variable? |
| Variable violations | How many of those rules are currently broken? |
| Template percentage | What percentage of its rules are broken? |
| Current violations | How many rules are broken in the whole solution? |
| Current satisfaction | How many rules are satisfied in the whole solution? |
| Number of constraints | How many rules exist altogether? |

We excluded the actual improvement from the inputs. Giving the model that answer would be like giving a student the answer key during an exam.

### Step 3 — Separate learning, model selection, and final testing

We divided the problem instances into three groups:

| Group | Instances | Purpose |
|---|---:|---|
| Fit | 128 | Teach the models |
| Validation | 32 | Compare settings and choose the model |
| Test | 40 | Measure final performance on unseen problems |

All examples from the same problem stayed together. This prevents closely related examples from appearing on both sides of an evaluation.

Think of these as **practice questions, a practice exam, and a final exam**.

### Step 4 — Explore the training data

We checked for missing values and duplicate records, examined the numbers, and created graphs.

One useful finding was that **48.1% of training candidates produced no improvement**. In other words, changing a randomly considered variable often does not help.

We also found that higher violation percentages generally relate to larger gains, but the relationship is imperfect.

### Step 5 — Train a Decision Tree

A **Decision Tree** learns a sequence of questions about the inputs, such as whether a variable has many violations. At the end of each branch, it predicts an improvement.

We tried **12 combinations of settings** controlling how complex the tree could become. We selected the combination with the lowest validation error.

### Step 6 — Train a Random Forest

A **Random Forest** combines predictions from many Decision Trees. Averaging their predictions can make the result more stable.

We tried **8 combinations of settings**. The selected forest used **200 trees**.

### Step 7 — Compare prediction accuracy and choose a model

We measured **Mean Absolute Error**, or **MAE**. It means:

> “On average, how far is the predicted improvement from the real improvement?”

Lower is better.

| Model | Validation MAE |
|---|---:|
| Decision Tree | 0.5183 |
| Random Forest | 0.5148 |

The Random Forest was slightly better, so we selected it. We made that choice **before checking the final test results**.

We then trained both chosen configurations again using all 160 available training instances.

### Step 8 — Test prediction accuracy on unseen problems

The final test MAEs were:

| Model | Test MAE |
|---|---:|
| Decision Tree | 0.4439 |
| Random Forest | 0.4365 |

The forest’s predictions were off by about **0.44 violations per candidate on average**. This does not mean “44% wrong.”

### Step 9 — Test the question that actually matters

Accurate predictions are useful only if they lead to good choices.

For each of the **200 test states**:

1. The simple percentage rule chose a variable.
2. The Random Forest predicted the improvement for all eight variables and chose its highest prediction.
3. Each method started from the **same original assignment**.
4. Both used the **same procedure** to change their chosen variable.
5. We compared the actual reductions in violations.

This kept the comparison fair: the variable-selection method was what changed.

## 6. What we found

| Outcome | Percentage rule | ML selection |
|---|---:|---:|
| Average violations removed per state | 1.565 | 1.690 |

ML removed **0.125 more violations per state on average**, roughly **8% more than the rule’s average improvement**.

Across the 200 states:

- ML did better on **23**.
- Both achieved equal improvement on **172**.
- ML did worse on **5**.

So the result is a **small average improvement, with most outcomes tied**. It does not mean ML was better on every problem, or that a complete Genetic Algorithm became 8% better.

## 7. What we produced and checked

We created:

- An executed notebook containing the code, results, graphs, and explanations.
- An eight-page report.
- A nine-slide presentation with speaker notes.
- Supporting data, results, and instructions for running the experiment.

We ran the notebook from beginning to end and independently checked the split, comparison, and calculations.

The team contribution statement still needs your members’ actual names and contributions. The worked examples and extra analysis suggested in the review are improvements we have identified; they have not yet been added.

The central lesson is that **predicting a number accurately and making a useful decision are different things**. This assignment required us to measure both.
