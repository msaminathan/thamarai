import streamlit as st
from tinyec import registry
from tinyec.ec import Point as ECCPoint

st.set_page_config(
    page_title="Step-by-Step Example",
)

st.title("🔬 Step-by-Step Encryption Example")

st.header("The Goal")
st.write(
    "Let's walk through the encryption and decryption of a single character, 'A', from Alice to Bob. "
    "We'll see the exact values at each step."
)

st.warning(
    "This example uses a very small, insecure elliptic curve for clarity. **Do not use this in production.**"
)

# Use a small, fixed curve for predictability
curve_name = "secp192r1"
curve = registry.get_curve(curve_name)
G = curve.g

# Alice and Bob's fixed keys
d_alice = 55  # Alice's private key
Q_alice = d_alice * G
d_bob = 99    # Bob's private key
Q_bob = d_bob * G

st.subheader("Step 1: Key Exchange")
st.write(
    "Before any communication, Alice and Bob generate their key pairs and exchange public keys. "
    "Only Bob's public key is needed for encryption."
)

st.code(
    """
# Bob's Private Key
d_bob = 99

# Bob's Public Key (Q_bob = d_bob * G)
Q_bob = 99 * G
""",
    language="python"
)

st.write(f"Bob's Private Key ($d_{{Bob}}$): **{d_bob}**")
st.write(f"Bob's Public Key ($Q_{{Bob}}$): **({Q_bob.x}, {Q_bob.y})**")

st.markdown("---")

st.subheader("Step 2: Alice Prepares the Message")
st.write(
    "Alice wants to send the message 'A'. She must first convert this character into a point on the curve. "
    "A simple (though insecure) way is to use the ASCII value."
)

message_char = 'A'
message_value = ord(message_char)
message_point = message_value * G

st.code(
    """
message_char = 'A'
message_value = ord(message_char)  # ASCII value is 65
message_point = message_value * G
""",
    language="python"
)

st.write(f"Message Character: **'{message_char}'**")
st.write(f"ASCII Value: **{message_value}**")
st.write(f"Message Point ($P_m$): **({message_point.x}, {message_point.y})**")

st.markdown("---")

st.subheader("Step 3: Alice Encrypts the Message")
st.write(
    "Alice uses Bob's public key and a temporary, random key ($k$) to create a ciphertext consisting of two points, $C_1$ and $C_2$."
)

k_alice = 37  # Alice's ephemeral private key for this message
st.code(
    """
# Alice's ephemeral key
k_alice = 37

# Calculate C1
C1 = k_alice * G

# Calculate C2
C2 = message_point + (k_alice * Q_bob)
""",
    language="python"
)

# Step-by-step calculation output
C1 = k_alice * G
k_times_Q_bob = k_alice * Q_bob
C2 = message_point + k_times_Q_bob

st.write("Alice calculates the first point:")
st.write(f"$$C_1 = k_{{Alice}} \\times G = 37 \\times G$$")
st.write(f"**$C_1$ = ({C1.x}, {C1.y})**")
st.write("")

st.write("Alice calculates the second point:")
st.write(f"$$C_2 = P_m + (k_{{Alice}} \\times Q_{{Bob}})$$")
st.write(f"First, she computes the shared secret: $k_{{Alice}} \\times Q_{{Bob}}$")
st.write(f"  - Shared Secret = $37 \\times ({Q_bob.x}, {Q_bob.y})$")
st.write(f"  - Shared Secret = **({k_times_Q_bob.x}, {k_times_Q_bob.y})**")
st.write("")
st.write(f"Then she adds the message point to it:")
st.write(f"  - $C_2 = ({message_point.x}, {message_point.y}) + ({k_times_Q_bob.x}, {k_times_Q_bob.y})$")
st.write(f"**$C_2$ = ({C2.x}, {C2.y})**")

st.success("Alice sends the ciphertext **$(C_1, C_2)$** to Bob.")

st.markdown("---")

st.subheader("Step 4: Bob Decrypts the Message")
st.write(
    "Bob receives the two points and uses his secret private key ($d_{{Bob}}$) to recover the original message point. "
    "He does this by calculating the same shared secret that Alice did."
)

st.code(
    """
# Bob receives C1 and C2
# He uses his private key to find the shared secret
shared_secret = d_bob * C1

# Decrypt the message point
decrypted_point = C2 - shared_secret
""",
    language="python"
)

# Step-by-step calculation output
bob_shared_secret = d_bob * C1
decrypted_point = C2 - bob_shared_secret

st.write("Bob first computes the shared secret:")
st.write(f"$$d_{{Bob}} \\times C_1 = 99 \\times ({C1.x}, {C1.y})$$")
st.write(f"**Shared Secret = ({bob_shared_secret.x}, {bob_shared_secret.y})**")
st.info("Notice that this is the **exact same** shared secret Alice calculated in Step 3!")

st.write("Bob then subtracts this shared secret from $C_2$ to isolate the message point:")
st.write(f"$$P_m = C_2 - (d_{{Bob}} \\times C_1)$$")
st.write(f"$$P_m = ({C2.x}, {C2.y}) - ({bob_shared_secret.x}, {bob_shared_secret.y})$$")
st.write(f"**$P_m$ = ({decrypted_point.x}, {decrypted_point.y})**")

st.markdown("---")

st.subheader("Step 5: Bob Converts the Point Back to a Character")
st.write(
    "Finally, Bob converts the decrypted point back into its original character. "
    "This is simply the reverse of the mapping in Step 2."
)

decrypted_value = int(decrypted_point.x / G.x)  # Simple reverse mapping
decrypted_char = chr(decrypted_value)

st.code(
    """
decrypted_value = int(decrypted_point.x / G.x)
decrypted_char = chr(decrypted_value)
""",
    language="python"
)

st.success(f"The decrypted value is **{decrypted_value}**, which corresponds to the character **'{decrypted_char}'**.")
