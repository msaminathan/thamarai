import streamlit as st
import random
from tinyec.ec import Point
from tinyec import registry
from tinyec.ec import Point as ECCPoint

st.set_page_config(
    page_title="Example",
)

st.title("👨‍🔬 A Full Example: Encrypting a Message")

st.header("1. The Players: Alice and Bob")
st.write(
    "Let's see how Alice sends a secret message to Bob using ECC. "
    "We'll use a small, insecure curve for this demonstration."
)

curve = registry.get_curve("secp192r1")

# Bob's Key Generation (Simulated)
st.subheader("Bob's Keys")
d_bob = 54  # Bob's private key
Q_bob = d_bob * curve.g  # Bob's public key
st.info(f"Bob's Private Key ($d_{{Bob}}$): **{d_bob}**")
st.success(f"Bob's Public Key ($Q_{{Bob}}$): **({Q_bob.x}, {Q_bob.y})**")

st.header("2. Alice's Encryption Process")
st.write(
    "Alice gets Bob's public key from a public directory. Now she wants to send the message **'Hello World'**."
)

st.subheader("Step 2.1: Convert Message to Points")

# A very simple mapping: We'll take each character and represent it as an integer, then a point
def char_to_point(char, curve, base_point):
    val = ord(char)
    # A simple, illustrative (not secure) way to get a point
    return val * base_point

message = "Hello World"
message_points = [char_to_point(c, curve, curve.g) for c in message]
st.write(f"Original Message: **'{message}'**")
st.write("The message is converted into a list of points on the curve:")
for i, p in enumerate(message_points):
    st.write(f"  - Point {i}: **({p.x}, {p.y})**")

st.subheader("Step 2.2: Generate Ephemeral Key and Encrypt")
st.write(
    "For each message point, Alice generates a random ephemeral key ($k$) and calculates two points for the ciphertext."
)

k_alice = 37  # Alice's ephemeral private key
st.write(f"Alice's ephemeral private key ($k$): **{k_alice}**")

# Encryption loop
ciphertext = []
for p_m in message_points:
    C1 = k_alice * curve.g
    C2 = p_m + (k_alice * Q_bob)
    ciphertext.append((C1, C2))

st.write("Alice's ciphertext is a list of (C1, C2) pairs:")
for i, (c1, c2) in enumerate(ciphertext):
    st.write(f"  - Pair {i}: C1=({c1.x}, {c1.y}), C2=({c2.x}, {c2.y})")

st.info("Alice sends this ciphertext to Bob.")

st.header("3. Bob's Decryption Process")
st.write(
    "Bob receives the ciphertext and uses his private key to recover the original message."
)

st.subheader("Step 3.1: Decrypt Each Point")
decrypted_points = []
for c1, c2 in ciphertext:
    # Bob calculates the shared secret
    shared_secret = d_bob * c1
    # Bob subtracts the shared secret from C2 to get the message point
    p_m_decrypted = c2 - shared_secret
    decrypted_points.append(p_m_decrypted)

st.write("Bob decrypts each ciphertext pair to get the original message points:")
for i, p in enumerate(decrypted_points):
    st.write(f"  - Decrypted Point {i}: **({p.x}, {p.y})**")

st.subheader("Step 3.2: Convert Points Back to Message")

# A very simple mapping back to the character
def point_to_char(point, base_point):
    # This is a highly insecure, brute-force way for a demo
    # In a real-world scenario, you would have a secure mapping scheme
    val = int(point.x / base_point.x)
    return chr(val)

decrypted_message = "".join([point_to_char(p, curve.g) for p in decrypted_points])

st.success(f"Bob successfully decrypts the message: **'{decrypted_message}'**")
