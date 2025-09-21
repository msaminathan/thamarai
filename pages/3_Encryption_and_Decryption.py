import streamlit as st
from tinyec.ec import Point
from tinyec import registry

st.set_page_config(
    page_title="Encryption & Decryption",
)

st.title("🔒 Encryption and Decryption")

st.header("The Process (Elliptic Curve Integrated Encryption Scheme)")
st.write(
    "Let's say Alice wants to send a message to Bob. The following steps are taken:"
)

# Use a small curve for the example
curve = registry.get_curve("secp192r1")

st.subheader("1. Setup")
d_bob = 27  # Bob's private key
Q_bob = d_bob * curve.g  # Bob's public key
st.write(f"Bob's Private Key ($d_{{Bob}}$): **{d_bob}**")
st.write(f"Bob's Public Key ($Q_{{Bob}}$): **({Q_bob.x}, {Q_bob.y})**")

k = 50  # Alice's ephemeral private key
P_m = 10 * curve.g  # A dummy message point
st.write(f"Alice's ephemeral private key ($k$): **{k}**")
st.write(f"The message point ($P_m$): **({P_m.x}, {P_m.y})**")

st.subheader("2. Encryption (by Alice)")
st.write(
    "Alice uses Bob's public key to create a ciphertext, which consists of two points:"
)
st.latex(r"C_1 = k \times G")
C1 = k * curve.g
st.write(f"Point C1: **({C1.x}, {C1.y})**")

st.latex(r"C_2 = P_m + (k \times Q_{Bob})")
k_times_Q_bob = k * Q_bob
C2 = P_m + k_times_Q_bob
st.write(f"Point C2: **({C2.x}, {C2.y})**")

st.write(f"Alice sends the ciphertext $(C_1, C_2)$ to Bob.")

st.subheader("3. Decryption (by Bob)")
st.write(
    "Bob receives $(C_1, C_2)$ and uses his private key to recover the original message."
)
st.write("Bob first calculates a shared secret point:")
st.latex(r"d_{Bob} \times C_1 = d_{Bob} \times (k \times G) = k \times (d_{Bob} \times G) = k \times Q_{Bob}")
shared_secret = d_bob * C1
st.write(f"Shared Secret: **({shared_secret.x}, {shared_secret.y})**")

st.write("Bob then recovers the message point:")
st.latex(r"P_m = C_2 - (d_{Bob} \times C_1)")
decrypted_P_m = C2 - shared_secret
st.write(f"Decrypted Message Point ($P_m$): **({decrypted_P_m.x}, {decrypted_P_m.y})**")

st.write("The decrypted point matches the original message point. The encryption is successful! ✅")
