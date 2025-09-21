import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ECC vs RSA",
)

st.title("⚖️ ECC vs. RSA")

st.header("1. Fundamental Difference")
st.write(
    "Both ECC and RSA are public-key cryptosystems, but their security models are based on different "
    "mathematical problems:"
)
st.write(
    "- **RSA:** Security is based on the difficulty of **factoring large prime numbers**."
)
st.write(
    "- **ECC:** Security is based on the difficulty of the **Elliptic Curve Discrete Logarithm Problem (ECDLP)**."
)

st.header("2. Key Size Comparison")
st.write(
    "The ECDLP is a much harder problem to solve than the integer factorization problem for the same key size. "
    "This means ECC can provide the same level of security with significantly smaller keys."
)

data = {
    "Symmetric Security (bits)": [80, 112, 128, 192, 256],
    "ECC Key Size (bits)": [160, 224, 256, 384, 512],
    "RSA Key Size (bits)": [1024, 2048, 3072, 7680, 15360]
}
df = pd.DataFrame(data).set_index("Symmetric Security (bits)")
st.table(df)

st.write(
    "As you can see, a 256-bit ECC key offers equivalent security to a 3072-bit RSA key. "
    "This is ECC's biggest advantage."
)

st.header("3. Performance and Resource Efficiency")
st.write(
    "Because of the smaller key size, ECC has significant performance advantages:"
)
st.write("- **Faster computation:** Key generation and digital signatures are much faster.")
st.write("- **Less storage:** Smaller keys require less memory and disk space.")
st.write("- **Lower bandwidth:** Less data needs to be transmitted for key exchange and signatures.")

st.write(
    "This makes ECC the ideal choice for resource-constrained environments like mobile devices, "
    "smart cards, and IoT devices."
)

st.header("4. When to use each?")
st.write(
    "Today, ECC is widely used in TLS/SSL, cryptocurrencies (Bitcoin, Ethereum), and many modern security protocols. "
    "RSA is still in use, especially in legacy systems, but ECC is now the preferred standard for new implementations due to its efficiency and comparable security."
)
