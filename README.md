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
