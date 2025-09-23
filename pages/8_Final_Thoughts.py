import streamlit as st

st.set_page_config(
    page_title="Final Thoughts",
)

st.title("💡 Final Thoughts and Deeper Insights")

st.header("1. The Core Wisdom of ECC")
st.write(
    "The true wisdom of ECC isn't just in the complex math; it's in its elegant efficiency. "
    "By moving from a multiplication problem over integers (like in RSA) to a **point multiplication problem on a curve**, "
    "ECC found a way to create a 'trapdoor' function that is incredibly difficult to reverse, but with far less computational effort."
)

st.subheader("The Trapdoor Function Analogy")
st.write(
    "Think of ECC like a safe with a very specific, one-way combination lock. "
    "Your **private key** is the secret combination number. "
    "Your **public key** is the final, open state of the safe's dial after you've spun it many times."
)
st.write(
    "It's easy for you (and anyone with the private key) to quickly spin the dial to its final state, "
    "but it's virtually impossible for someone who only sees the final dial position to figure out how many times you spun it."
)
st.image("https://upload.wikimedia.org/wikipedia/commons/e/e0/Public_key_cryptography_diagram.svg",
         caption="The public key is derived from the private key, but not vice-versa.")

st.markdown("---")

st.header("2. Where is ECC Today?")
st.write(
    "ECC is not just a theoretical concept; it's the backbone of modern digital security. You've likely used it today without even knowing it."
)

st.subheader("Real-World Applications")
st.markdown(
    """
    - **🌐 HTTPS/TLS:** Most modern websites use ECC certificates for secure connections.
    - **₿ Blockchain:** Bitcoin, Ethereum, and other cryptocurrencies use the **Elliptic Curve Digital Signature Algorithm (ECDSA)** to verify transactions. Your wallet address is a hashed version of your public key.
    - **📱 Mobile Devices:** Due to their smaller key sizes and lower power consumption, ECC is perfect for securing data on smartphones and smart cards.
    - **🔒 Secure Messaging:** Apps like Signal and WhatsApp use ECC to provide end-to-end encryption.
    """
)

st.markdown("---")

st.header("3. The Looming Threat of Quantum Computing")
st.write(
    "While ECC is secure against today's computers, the rise of large-scale **quantum computers** presents a future threat."
)
st.write(
    "An algorithm called **Shor's algorithm** could theoretically solve the elliptic curve discrete logarithm problem (and RSA's integer factorization problem) in a fraction of the time, breaking ECC security."
)
st.write(
    "This has led to the development of **Post-Quantum Cryptography (PQC)**, which aims to create new, quantum-resistant algorithms. The cryptographic world is already preparing for the transition to these new standards."
)
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Shor%27s_algorithm_on_a_quantum_computer.svg/500px-Shor%27s_algorithm_on_a_quantum_computer.svg.png",
         caption="Shor's algorithm could break current public-key cryptography.")

st.markdown("---")

st.header("4. Conclusion")
st.write(
    "You've taken the first step on a fascinating journey into modern cryptography. "
    "ECC is a beautiful blend of abstract mathematics and practical application. "
    "Understanding its core principles gives you a profound insight into how our digital world is secured. Keep exploring!"
)
