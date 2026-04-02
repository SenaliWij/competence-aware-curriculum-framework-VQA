# Competence-Aware Curriculum Framework

This document explains the internal flow of the curriculum folder, detailing how the model progresses through the difficulty tiers and the impact of the key hyperparameters controlling its learning trajectory.

---

## 1. High-Level Flow

The curriculum framework controls **what data the model sees** when training. Rather than feeding it samples randomly across all difficulty levels (Tiers 1-5), the framework estimates how "competent" the model currently is and shifts the probability of training on harder tiers as the model improves. 

The primary training loop is orchestrated by `CurriculumTrainer` in `training/curriculum_trainer.py`. Here's exactly what happens during each step of the training loop:

1. **Get Current Competence**: Read the model's overall competence score `C` from the `CompetenceTracker` (ranging from 0.05 to 0.99).
2. **Tier Sampling**: Use the competence score and the defined difficulty of each tier to assign a probability distribution across tiers. A tier is randomly drawn based on these weights.
3. **Soft Self-Paced Learning (SPL) Sampling**: Within the drawn tier, compute a temperature ($\lambda$) based on competence to select a batch of samples. (Easier samples in the tier are favored initially; as `C` goes up, harder samples are favored).
4. **Train and Predict**: The batch is passed through the model. The model computes the training loss and predictions (logits), from which we calculate the entropy.
5. **Update Competence**: The batch's entropy and loss are fed back into the `CompetenceTracker`, adjusting the model's global competence up or down via an Exponential Moving Average (EMA).

---

## 2. Core Components and Code Snippets

### The Main Training Loop (`CurriculumTrainer`)

Located in `training/curriculum_trainer.py`, the `train()` method holds the core sequence of events:

```python
for step in range(start_step, self.num_steps):
    # 1) Current competence
    competence = self.tracker.competence()

    # 2) Tier probabilities & sample a tier
    tier_probs = self.tier_sampler.tier_probabilities(competence)
    tier = self.tier_sampler.sample_tier(competence)

    # ... SPL index sampling omitted for brevity ...

    # 4) Build a mini-batch from the selected indices
    batch = next(iter(loader))

    # 5) Forward pass to get logits (for entropy) & loss
    fwd = self.model.forward_step(batch)
    h_batch = batch_entropy(fwd["logits"])
    l_batch = fwd["loss"]

    # 6) Update EMA tracker
    self.tracker.update(h_batch, l_batch)

    # 8) Actual training step (backprop)
    train_out = self.model.train_step(batch)
```

### Competence Tracking (`CompetenceTracker`)

Located in `services/competence_tracker.py`, the tracker calculates a single scalar `C`. It is a combination of how **confident** the model is (entropy component) and how **accurate** the model is (loss component).

```python
    def update(self, h_batch: float, l_batch: float) -> None:
        self.h_ema = self.beta * self.h_ema + (1 - self.beta) * h_batch
        self.l_ema = self.beta * self.l_ema + (1 - self.beta) * l_batch
        self.step_count += 1

    def competence(self) -> float:
        # ... Bias corrections omitted ...
        
        # Normalised entropy component (1.0 = zero entropy / high confidence)
        entropy_term = 1.0 - (h_ema / self.h_max)
        
        # Normalised loss component (1.0 = zero loss / high accuracy)
        loss_term = 1.0 - (l_ema / self.l0)

        # Final Competence Score
        c = self.entropy_weight * entropy_term + self.loss_weight * loss_term
        return max(self.c_min, min(self.c_max, c))
```

### Tier Assignment Matrix (`TierSampler`)

Located in `services/tier_sampler.py`, the probability of picking tier `k` is based on a power law equation:
$Score(k) = C^{difficulty[k]}$

As `difficulty` increases (Tier 5 has a much higher exponent than Tier 1), $C^{difficulty}$ approaches zero faster because $C < 1$. High tiers therefore remain exceedingly rare until the competence `C` is very close to `1.0`.

```python
    def tier_scores(self, competence: float) -> Dict[int, float]:
        # C ^ difficulty[k]
        return {k: competence ** d for k, d in self.difficulty.items()}
```

---

## 3. The Impact of Important Parameters

### What a lower `beta` (EMA Value) would do
The EMA value (`beta`, typically `0.9`) acts as a shock absorber. 
$$EMA_{new} = \beta \cdot EMA_{old} + (1 - \beta) \cdot batch\_value$$

- **If you lower it (e.g., to 0.5 or 0.1)**: The tracker will "forget" history quickly and weigh the current batch heavily. Competence will jump wildly up and down based on the ease of the current batch. The curriculum will progress very erratically—a few lucky easy batches might instantly shoot competence to 0.99, throwing the model into the hardest tier before it's ready.
- **If you raise it (e.g., to 0.999)**: The model's competence will climb extremely smoothly but very slowly, ignoring sporadic bad batches.

### What a lower `entropy_weight` would do
The total competence is a weighted average between:
$$C = \alpha \cdot (Entropy Component) + (1-\alpha) \cdot (Loss Component)$$
*(Where `entropy_weight` is $\alpha$ and `loss_weight` is $1-\alpha$)*

- **Entropy** simply tracks the model's statistical confidence (is the softmax distribution sharp?). An untrained model can quickly learn to be confidently wrong (sharp distributions on wrong answers), giving it a false high competence.
- **Loss** tracks if the model is actually getting the answers right.

**If you lower the `entropy_weight` (which naturally raises the `loss_weight`)**:
The curriculum forces the model to actually be *accurate* to progress, regardless of how confident its predictions are. The competence score will primarily climb only when the model learns to answer questions correctly (lower loss). This prevents premature progression where the model confidently guesses "Yes" for everything and triggers a jump to harder tiers without actually understanding the problem.

### What the Initial Bias Correction does
When the tracking begins, `h_ema` and `l_ema` contain zero history. During early steps (the first `bias_correction_steps`), the framework scales up the EMA to counteract this lack of history. 
```python
correction = 1.0 - (self.beta ** self.step_count)
```
This is a standard machine learning technique (adopted from the Adam Optimizer) that ensures competence doesn't start artificially inflated (or deflated) during the first few dozen training steps while the EMA "warms up".

### What the Tier Difficulty Exponents do
Defined in `main.py` as `DEFAULT_DIFFICULTY = {1: 1.0, 2: 3.0, 3: 5.0, 4: 7.0, 5: 10.0}`.
Since competence $0 < C < 1$, raising it to a higher power shrinks the score dramatically. 

- At $C = 0.5$, Tier 1's score is $0.5^{1} = 0.5$, while Tier 5's score is $0.5^{10} = 0.00097$. The sampler almost exclusively selects from Tier 1.
- As $C$ approaches $0.9$, Tier 5's probability mass starts increasing enough to be repeatedly sampled. Changing Tier 5's exponent from 10.0 to 15.0 would require the model to exhibit near-perfect competence across Tiers 1-4 before attempting Tier 5.
