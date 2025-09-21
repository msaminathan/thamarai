import streamlit as st
from tinyec.ec import Point, Curve
from tinyec import registry

st.set_page_config(
    page_title="Key Generation",
)

st.title("🔑 Key Generation")

st.header("1. The Secret and The Public")
st.write(
    "In ECC, each user generates a pair of keys:"
)
st.write(
    "- A **private key ($d_A$)**: A randomly chosen large integer. This must be kept secret."
)
st.write(
    "- A **public key ($Q_A$)**: A point on the elliptic curve. This is calculated by multiplying the "
    "base point $G$ by the private key: $Q_A = d_A \\times G$. This key can be shared publicly."
)

st.header("2. The Base Point ($G$) and Curve Parameters")
st.write(
    "The entire system depends on a publicly agreed-upon **base point** $G$ on a specific elliptic curve. "
    "For this example, we will use a small, pre-defined curve for demonstration purposes."
)

st.header("3. The Illustrative Example")

# Use a small curve for the example
curve_name = st.selectbox(
    "Select a curve for the demonstration:",
    options=["secp256r1", "secp192r1"],
    index=0
)
curve = registry.get_curve(curve_name)

st.subheader(f"Using {curve_name} Curve")
st.write(f"The base point for this curve is G = ({curve.g.x}, {curve.g.y})")

# Generate a random private key for Alice
d_alice = st.slider("Select Alice's private key (a random integer):", 1, 100)
Q_alice_x, Q_alice_y = (d_alice * curve.g).x, (d_alice * curve.g).y

st.write(f"Alice's Private Key ($d_{{Alice}}$): **{d_alice}**")
st.write(f"Alice's Public Key ($Q_{{Alice}}$) = ${d_alice} \\times G$")
st.write(f"Alice's Public Key ($Q_{{Alice}}$): **({Q_alice_x}, {Q_alice_y})**")

st.header("4. The Discrete Logarithm Problem")
st.write(
    "The security of ECC relies on the **Elliptic Curve Discrete Logarithm Problem (ECDLP)**. "
    "Given the base point $G$ and the public key $Q_A$, it is computationally infeasible "
    "to find the private key $d_A$."
)
st.write(
    "While we can easily calculate $Q_A = d_A \\times G$, reversing the process to find $d_A$ from $Q_A$ is "
    "extremely difficult for large numbers."
)
