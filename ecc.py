import streamlit as st

st.set_page_config(
    page_title="Elliptic Curve Cryptography",
    page_icon="🔐",
)

st.title("🔐 Elliptic Curve Cryptography (ECC) Explainer")
st.sidebar.success("Select a page above.")

st.markdown(
    """
    Welcome to this interactive guide on Elliptic Curve Cryptography (ECC)!

    ECC is a powerful public-key cryptographic system used to secure data in modern applications,
    from web browsing to cryptocurrency. This app will walk you through the core concepts,
    the mathematics, and its practical use.

    **Select a topic from the sidebar to begin.**
    """
)

st.header("What is ECC?")
st.write(
    "ECC is a public-key cryptography system based on the algebraic structure of elliptic curves over finite fields. "
    "Unlike RSA, which relies on the difficulty of factoring large numbers, ECC's security is based on the "
    "**elliptic curve discrete logarithm problem (ECDLP)**. This makes it highly efficient, offering "
    "the same level of security with much smaller key sizes."
)
