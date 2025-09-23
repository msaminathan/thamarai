import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Theory",
)

st.title("📜 The Theory Behind ECC")

st.header("1. The Elliptic Curve Equation")
st.write(
    "An elliptic curve is not an ellipse. It's a specific type of curve defined by the equation:"
)
st.latex(r"y^2 = x^3 + ax + b")

st.write(
    "In cryptography, we use these curves over a **finite field**, which means all the points have "
    "integer coordinates, and all calculations are performed modulo a large prime number $p$. "
    "This transforms the smooth curve into a scatter plot of discrete points."
)

st.header("2. Point Addition over a Finite Field")
st.write("Let's visualize an elliptic curve over a small finite field, and then perform point addition.")

# Define a small, illustrative curve
p = 13  # A prime number
a = 1
b = 1
st.write(f"Using the curve $y^2 \\equiv x^3 + {a}x + {b} \\pmod{{{p}}}$")

# Find all points on the curve
points = []
for x in range(p):
    rhs = (x**3 + a*x + b) % p
    for y in range(p):
        lhs = (y**2) % p
        if lhs == rhs:
            points.append((x, y))

# Plot the points
fig, ax = plt.subplots(figsize=(6, 6))
x_coords = [p[0] for p in points]
y_coords = [p[1] for p in points]
ax.plot(x_coords, y_coords, 'o', markersize=10, color='blue', label='Points on the Curve')
ax.set_title(f"Elliptic Curve over Finite Field $GF({p})$")
ax.set_xlabel(f"x (mod {p})")
ax.set_ylabel(f"y (mod {p})")
ax.set_xticks(range(p))
ax.set_yticks(range(p))
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.markdown("---")

st.header("3. Formulas and Example")
st.write(
    "The formulas for point addition and doubling over a finite field are the same as in the real number plane, "
    "but all divisions are replaced with modular inverse operations."
)

st.subheader("Formulas")
st.write("Given two points $P = (x_P, y_P)$ and $Q = (x_Q, y_Q)$, the resulting point is $R = (x_R, y_R)$.")
st.markdown("**If $P \\neq Q$ (Point Addition):**")
st.latex(r"m \equiv \frac{y_Q - y_P}{x_Q - x_P} \pmod{p}")
st.latex(r"x_R \equiv m^2 - x_P - x_Q \pmod{p}")
st.latex(r"y_R \equiv m(x_P - x_R) - y_P \pmod{p}")

st.markdown("**If $P = Q$ (Point Doubling):**")
st.latex(r"m \equiv \frac{3x_P^2 + a}{2y_P} \pmod{p}")
st.latex(r"x_R \equiv m^2 - 2x_P \pmod{p}")
st.latex(r"y_R \equiv m(x_P - x_R) - y_P \pmod{p}")

st.write(
    "The modular inverse is a number $n^{-1}$ such that $n \\times n^{-1} \\equiv 1 \\pmod{p}$. "
    "It's like division in modular arithmetic."
)

st.subheader("Example: Point Addition on the Plot")
st.write("Let's add two points from our curve: $P = (1, 1)$ and $Q = (3, 7)$.")

# Find modular inverse function
def mod_inverse(n, p):
    for i in range(p):
        if (n * i) % p == 1:
            return i
    return None

# Step-by-step calculation
st.write("### Calculation Steps")

st.markdown("**Step 1: Calculate the slope $m$**")
st.code(
    """
y_Q - y_P = 7 - 1 = 6
x_Q - x_P = 3 - 1 = 2
mod_inverse(2, 13) = 7  (since 2 * 7 = 14 = 1 mod 13)
m = 6 * 7 = 42
m mod 13 = 3
""", language="text"
)
m = 3

st.markdown("**Step 2: Calculate the x-coordinate of the result**")
st.code(
    """
x_R = m^2 - x_P - x_Q
x_R = 3^2 - 1 - 3 = 9 - 4 = 5
x_R mod 13 = 5
""", language="text"
)
x_R = 5

st.markdown("**Step 3: Calculate the y-coordinate of the result**")
st.code(
    """
y_R = m * (x_P - x_R) - y_P
y_R = 3 * (1 - 5) - 1
y_R = 3 * (-4) - 1 = -12 - 1 = -13
y_R mod 13 = 0
""", language="text"
)
y_R = 0

st.write(f"The resulting point is **$R = P+Q = ({x_R}, {y_R})$**.")

# Plot the points on the graph
fig, ax = plt.subplots(figsize=(6, 6))
x_coords = [p[0] for p in points]
y_coords = [p[1] for p in points]
ax.plot(x_coords, y_coords, 'o', markersize=10, color='blue', label='Points on the Curve')
ax.plot([1, 3, 5], [1, 7, 0], 'o', markersize=12)
ax.plot(1, 1, 'go', markersize=12, label='P')
ax.plot(3, 7, 'ro', markersize=12, label='Q')
ax.plot(5, 0, 'yo', markersize=12, label='P+Q')
ax.set_title(f"Point Addition on $GF({p})$")
ax.set_xlabel(f"x (mod {p})")
ax.set_ylabel(f"y (mod {p})")
ax.set_xticks(range(p))
ax.set_yticks(range(p))
ax.grid(True)
ax.legend()
st.pyplot(fig)
