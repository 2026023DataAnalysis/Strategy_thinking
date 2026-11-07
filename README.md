### 3.1 From the Boltzmann Distribution to Maxwell-Boltzmann

**Step 1 – The Boltzmann distribution.**
Statistical mechanics says that, in thermal equilibrium at temperature $T$, the probability that a system is in a state with energy $E_i$ is

$$P(E_i) = \frac{1}{Z}\, e^{-E_i / k_B T}$$

- The factor $e^{-E_i/k_BT}$ is the **Boltzmann factor**. States with high energy are **less probable**, because the exponential becomes very small when $E_i \gg k_BT$.
- $k_BT$ is the "typical energy" that the surroundings can give to a molecule. A higher temperature makes high-energy states more accessible.
- $Z = \sum_i e^{-E_i/k_BT}$ is the **partition function**. It is a normalisation constant, so that all probabilities add up to 1.

**Step 2 – Apply it to one gas molecule.**
In an **ideal gas** the molecules do not attract or repel each other. The only energy that depends on the motion is the translational kinetic energy:

$$E_k = \tfrac{1}{2} m \left(v_x^2 + v_y^2 + v_z^2\right)$$

**Step 3 – Each velocity component is a normal distribution.**
Put $E_k$ into the Boltzmann factor. The exponential of a sum is a product of exponentials, so the three directions are **independent**:

$$f(v_x, v_y, v_z) \propto e^{-m v_x^2/2k_BT}\cdot e^{-m v_y^2/2k_BT}\cdot e^{-m v_z^2/2k_BT}$$

Each factor is a **normal (Gaussian) distribution** with mean 0 and variance $\sigma^2 = k_BT/m$. After normalisation:

$$f(v_x) = \sqrt{\frac{m}{2\pi k_BT}}\; e^{-m v_x^2/2k_BT}$$

This is a nice link to my ML course (lesson 1.3.4, continuous distributions): each **velocity component** of a gas molecule is a normal random variable.

**Step 4 – From velocity to speed.**
The speed is the length of the velocity vector: $v = \sqrt{v_x^2+v_y^2+v_z^2}$. Many different velocity vectors have the same speed $v$. In 3D velocity space, they all lie on a **sphere of radius $v$**, and a thin spherical shell has volume $4\pi v^2\,dv$. So we multiply the velocity probability by the shell area $4\pi v^2$:

$$f(v)\,dv = 4\pi v^2 \left(\frac{m}{2\pi k_BT}\right)^{3/2} e^{-m v^2/2k_BT}\,dv$$

This is the **Maxwell-Boltzmann speed distribution**.

**Connection to statistics.** If $v_x, v_y, v_z$ are independent normal variables with the same $\sigma$, then $v/\sigma$ follows a **chi distribution with 3 degrees of freedom**. In SciPy, this distribution is called `scipy.stats.maxwell`. That is why SciPy can generate our data directly (Section 4.2).

### 3.2 The Maxwell-Boltzmann Speed Distribution

$$\boxed{\,f(v) = 4\pi \left(\frac{m}{2\pi k_BT}\right)^{3/2} v^2\, \exp\!\left(-\frac{m v^2}{2k_BT}\right)\,}$$

**What does the equation represent?**
$f(v)$ is a **probability density**. The probability that a molecule has a speed between $v_1$ and $v_2$ is the area under the curve:

$$P(v_1 \le v \le v_2) = \int_{v_1}^{v_2} f(v)\,dv$$

The value $f(v)$ itself is *not* a probability. Its unit is s/m, so only $f(v)\,dv$ (an area) is a probability.

**The three parts of the equation**

| Part | Meaning |
|---|---|
| $4\pi\left(\frac{m}{2\pi k_BT}\right)^{3/2}$ | Normalisation constant. It makes the total area equal to 1. |
| $v^2$ | **Geometric factor.** The number of velocity directions with speed $v$ grows like the area of a sphere, $4\pi v^2$. It pushes probability **up** at higher speeds. |
| $\exp\left(-\frac{mv^2}{2k_BT}\right)$ | **Boltzmann factor.** High speed means high kinetic energy, and high energy is less probable. It pushes probability **down** at higher speeds. |

**Why is there a peak?** For small $v$, the $v^2$ term wins and $f(v)$ grows. For large $v$, the exponential wins and $f(v)$ goes to zero very fast. The competition between the two terms creates a single maximum, the **most probable speed**.

**Why is the distribution not symmetrical?**
- On the left, the speed cannot be negative, so the curve starts at $v = 0$ and rises like $v^2$.
- On the right, there is no upper limit, and the curve has a long **tail** of fast molecules.

So the distribution is **right-skewed** (positive skewness). This is why the mean is larger than the most probable speed. A normal distribution is always symmetric, so molecular speeds are **not normal**. This is the basis of hypothesis H2.

**What happens when the temperature increases?**
The temperature appears only in the combination $k_BT/m$. A higher $T$ gives a "wider" exponential, so:
1. The peak moves to **higher speeds** (molecules move faster on average).
2. The distribution becomes **wider** (more spread between slow and fast molecules).
3. The peak becomes **lower**. The total area must stay equal to 1, so a wider curve must be flatter.

This is exactly the "shift and flatten" behaviour that objective 2 asks us to show with EDA.

**Normalisation check (mathematics).** Let $\alpha = \dfrac{m}{2k_BT}$. Then $f(v) = 4\pi\left(\frac{\alpha}{\pi}\right)^{3/2} v^2 e^{-\alpha v^2}$. We use the standard Gaussian integral (ML course, lesson 1.2.6):

$$\int_0^\infty v^2 e^{-\alpha v^2}\,dv = \frac{\sqrt{\pi}}{4}\,\alpha^{-3/2}$$

$$\int_0^\infty f(v)\,dv = 4\pi\,\frac{\alpha^{3/2}}{\pi^{3/2}}\cdot\frac{\sqrt{\pi}}{4}\,\alpha^{-3/2} = \frac{4\pi\sqrt{\pi}}{4\pi^{3/2}} = 1 \;\checkmark$$

The total probability is exactly 1, as required for a PDF.

### 3.3 Characteristic Speeds: Most Probable, Mean and RMS

A distribution can be summarised by different "typical" values. For molecular speeds, physics uses three of them. We keep the notation $\alpha = m / 2k_BT$.

---

#### (a) Most probable speed $v_{mp}$: the peak of the curve

**Derivation.** The peak is where the derivative is zero (ML course, lessons 1.2.2 and 1.2.5). The constant does not change the position of the peak, so we only differentiate $g(v) = v^2 e^{-\alpha v^2}$ with the product rule:

$$g'(v) = 2v\,e^{-\alpha v^2} + v^2\,(-2\alpha v)\,e^{-\alpha v^2} = 2v\,e^{-\alpha v^2}\left(1 - \alpha v^2\right)$$

$g'(v) = 0$ for $v > 0$ only when $1 - \alpha v^2 = 0$, so $v^2 = 1/\alpha = 2k_BT/m$:

$$\boxed{v_{mp} = \sqrt{\frac{2k_BT}{m}}}$$

**Physical meaning.** If you pick one molecule, its speed is most likely to be near $v_{mp}$. It is the **mode** of the distribution.

---

#### (b) Mean speed $v_{mean}$: the average speed

**Derivation.** The mean is the expected value $E[v] = \int_0^\infty v\,f(v)\,dv$ (lesson 1.2.6, expectations). We need the integral $\int_0^\infty v^3 e^{-\alpha v^2}dv = \dfrac{1}{2\alpha^2}$ (substitute $u = v^2$):

$$v_{mean} = 4\pi\left(\frac{\alpha}{\pi}\right)^{3/2}\cdot\frac{1}{2\alpha^2} = \frac{2}{\sqrt{\pi\alpha}} = 2\sqrt{\frac{2k_BT}{\pi m}}$$

$$\boxed{v_{mean} = \sqrt{\frac{8k_BT}{\pi m}}}$$

**Physical meaning.** This is the arithmetic average of all molecular speeds. The **sample mean** of our simulated data should be very close to it (Law of Large Numbers, lesson 1.3.6). In Section 9.1, Linear Regression will try to learn this formula from data.

---

#### (c) Root-mean-square speed $v_{rms}$: linked to energy

**Derivation.** First we calculate the mean of $v^2$, using $\int_0^\infty v^4 e^{-\alpha v^2}dv = \dfrac{3\sqrt{\pi}}{8}\,\alpha^{-5/2}$:

$$\langle v^2\rangle = 4\pi\left(\frac{\alpha}{\pi}\right)^{3/2}\cdot\frac{3\sqrt{\pi}}{8}\,\alpha^{-5/2} = \frac{3}{2\alpha} = \frac{3k_BT}{m}$$

The RMS speed is the square root of this mean:

$$\boxed{v_{rms} = \sqrt{\langle v^2\rangle} = \sqrt{\frac{3k_BT}{m}}}$$

**Physical meaning.** $v_{rms}$ is the speed that gives the **average kinetic energy** (Section 3.4). Squaring gives more weight to fast molecules, so $v_{rms}$ is the largest of the three speeds.

---

#### Relationship with temperature

All three speeds have the same form, $v = \text{constant}\times\sqrt{T/m}$:

$$v \propto \sqrt{T}$$

- If the temperature is multiplied by 4, the speeds are only multiplied by $\sqrt{4} = 2$.
- The relationship between $v$ and $T$ is **not linear**. It is linear between $v$ and $\sqrt{T}$. This is very important for the Linear Regression in Section 9.1 (hypothesis H3).
- Heavier molecules are slower ($v \propto 1/\sqrt{m}$).

The ratio between the three speeds does **not** depend on temperature or mass:

$$v_{mp} : v_{mean} : v_{rms} = \sqrt{2} : \sqrt{8/\pi} : \sqrt{3} \approx 1 : 1.128 : 1.225$$

So the order is always $v_{mp} < v_{mean} < v_{rms}$. Two different reasons explain this order:
- $v_{mp} < v_{mean}$ because the distribution is **right-skewed**: the long tail of fast molecules pulls the mean to the right of the peak.
- $v_{mean} < v_{rms}$ is true for **any** distribution that is not a single value, because $\langle v^2\rangle - \langle v\rangle^2 = \sigma_v^2 > 0$ (the variance is positive).
