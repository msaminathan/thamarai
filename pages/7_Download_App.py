import streamlit as st
import base64
import os
import zipfile
import io

st.set_page_config(
    page_title="Download",
)

st.title("⬇️ Download the App")

st.header("Get the Complete Source Code")
st.write(
    "You can download the full source code for this multi-page Streamlit app. "
    "This includes all the Python files you've seen on the previous pages."
)

def create_zip(folder_path):
    """Creates a zip file from a folder in memory."""
    in_memory_zip = io.BytesIO()
    with zipfile.ZipFile(in_memory_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Walk through the folder
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to zip
                zf.write(file_path, os.path.relpath(file_path, folder_path))
    in_memory_zip.seek(0)
    return in_memory_zip.getvalue()

# Path to the pages folder
pages_folder = os.path.join(os.path.dirname(__file__), '..')

# Create a dummy zip file for demonstration (since we don't have a real file system)
dummy_zip_content = create_zip(pages_folder)
st.download_button(
    label="Download ECC App Code",
    data=dummy_zip_content,
    file_name="ecc_app.zip",
    mime="application/zip",
    help="Click to download the zip file containing all the app's code."
)

st.markdown("---")

st.header("Installation and Run Instructions")
st.write(
    "Once you have downloaded the `ecc_app.zip` file, follow these steps to run the application on your local machine."
)

st.subheader("Step 1: Unzip the File")
st.write(
    "Extract the contents of the `ecc_app.zip` file. You should see a folder named `ecc_app`."
)

st.subheader("Step 2: Install Dependencies")
st.write(
    "Open your terminal or command prompt and navigate to the `ecc_app` directory."
)
st.code(
    """
cd ecc_app
""",
    language="bash"
)
st.write("Then, install the required Python libraries using `pip`.")
st.code(
    """
pip install streamlit tinyec matplotlib
""",
    language="bash"
)

st.subheader("Step 3: Run the App")
st.write(
    "From the same terminal, run the following command to start the Streamlit application."
)
st.code(
    """
streamlit run app.py
""",
    language="bash"
)

st.write(
    "A new tab should automatically open in your web browser, and you will see the app running!"
)
