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
    "For this equation to be an elliptic curve, the discriminant must not be zero:"
)
st.latex(r"4a^3 + 27b^2 \neq 0")

st.write(
    "In cryptography, we use these curves over a **finite field**, which means all the points have "
    "integer coordinates, and all calculations are performed modulo a large prime number $p$. "
    "This transforms the smooth curve into a scatter plot of discrete points."
)

st.header("2. Point Addition and Multiplication")
st.write(
    "The 'magic' of ECC lies in its ability to perform 'point addition' and 'point multiplication'. "
    "These operations are well-defined and follow specific rules."
)

st.subheader("Interactive Plot: Point Addition and Doubling")

# Define the curve
a = -1
b = 1
y, x = np.ogrid[-2.5:2.5:0.01, -2.5:2.5:0.01]
#plt.style.use('dark_background')

# Create a plot for the curve y^2 = x^3 - x + 1
fig, ax = plt.subplots(figsize=(6, 6))
ax.contour(x.ravel(), y.ravel(), y**2 - (x**3 + a*x + b), [0])

# Sliders for interactive points
st.write("### Point Addition")
x_p = st.slider("Select x-coordinate for Point P:", -1.5, 1.5, 0.6)
y_p_sq = x_p**3 + a*x_p + b
y_p = np.sqrt(y_p_sq) if y_p_sq > 0 else 0
ax.plot(x_p, y_p, 'ro', label='P')

x_q = st.slider("Select x-coordinate for Point Q:", -1.5, 1.5, -1.0)
y_q_sq = x_q**3 + a*x_q + b
y_q = np.sqrt(y_q_sq) if y_q_sq > 0 else 0
ax.plot(x_q, y_q, 'go', label='Q')

# Logic for point addition (visual representation)
if x_p != x_q and y_p > 0 and y_q > 0:
    m = (y_q - y_p) / (x_q - x_p)
    x_r_temp = m**2 - x_p - x_q
    y_r_temp = y_p + m * (x_r_temp - x_p)
    ax.plot(x_r_temp, -y_r_temp, 'bo', label='P+Q')
    line_x = np.linspace(-2.5, 2.5, 100)
    line_y = m * (line_x - x_p) + y_p
    ax.plot(line_x, line_y, 'w--')
    ax.plot([x_r_temp, x_r_temp], [y_r_temp, -y_r_temp], 'y--')

ax.set_title("Elliptic Curve Point Operations")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.markdown(
    """
    ### **Point Addition (P + Q)**
    * Draw a straight line through points P and Q.
    * This line intersects the curve at a third point (in the code, this is `(x_r_temp, y_r_temp)`).
    * Reflect this third point across the x-axis to find the final point R = P + Q.
    """
)

st.markdown(
    """
    ### **Point Doubling (P + P = 2P)**
    * If you select two points very close to each other, the line becomes a **tangent** at the point P.
    * The tangent intersects the curve at one other point.
    * Reflecting this point across the x-axis gives the result of the addition, 2P.
    """
)
