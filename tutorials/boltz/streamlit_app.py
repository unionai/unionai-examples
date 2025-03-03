import streamlit as st
import tempfile
import os
import subprocess
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

st.title("Boltz Prediction")

uploaded_file = st.file_uploader("Upload YAML file", type=["yaml", "yml"])
msa_dir = st.file_uploader("Upload MSA directory (optional)", type=["zip", "tar", "gz"], accept_multiple_files=False)
options = st.text_area("Options (key=value format, one per line)")

USE_CPU_ONLY = os.environ.get("USE_CPU_ONLY", "0") == "1"
BOLTZ_MODEL = os.environ["BOLTZ_MODEL"]

logging.info(f"Contents of BOLTZ_MODEL directory ({BOLTZ_MODEL}):")
for root, dirs, files in os.walk(BOLTZ_MODEL):
    for name in files:
        logging.info(os.path.join(root, name))
    for name in dirs:
        logging.info(os.path.join(root, name))

if st.button("Predict"):
    logging.info("Predict button clicked")

    if uploaded_file is not None:
        logging.info("YAML file uploaded")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        logging.info(f"Temporary YAML file created at {tmp_file_path}")

        msa_dir_path = None
        if msa_dir is not None:
            logging.info("MSA directory uploaded")
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(msa_dir.name)[1]) as tmp_msa_file:
                tmp_msa_file.write(msa_dir.read())
                tmp_msa_file_path = tmp_msa_file.name
            msa_dir_path = tmp_msa_file_path
            logging.info(f"Temporary MSA directory created at {msa_dir_path}")

        options_list = ["--use_msa_server"]
        if options:
            logging.info("Options provided")
            for line in options.split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    options_list.append(f"--{key.strip()}={value.strip()}")
            logging.info(f"Options parsed: {options_list}")

        with tempfile.TemporaryDirectory() as out_dir:
            command = ["boltz", "predict", tmp_file_path, "--out_dir", out_dir, "--cache", BOLTZ_MODEL] + (["--accelerator", "cpu"] if USE_CPU_ONLY else []) + options_list
            if msa_dir_path:
                command += ["--msa_dir", msa_dir_path]
            logging.info(f"Running command: {' '.join(command)}")

            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                logging.info("Prediction completed successfully")
                st.success("Prediction completed successfully!")

                tar_path = shutil.make_archive(f"{out_dir}/boltz_results", 'gztar', out_dir)
                with open(tar_path, "rb") as tar_file:
                    st.download_button("Download Results", tar_file.read(), "boltz_results.tar.gz")
                logging.info(f"Results archived at {tar_path}")

            except subprocess.CalledProcessError as e:
                logging.error(f"Error during prediction: StdErr: {e.stderr}, StdOut: {e.stdout}")
                st.error(f"Error: {e.stderr}")

        # Clean up temporary files
        os.remove(tmp_file_path)
        logging.info(f"Temporary YAML file {tmp_file_path} removed")
        if msa_dir_path is not None:
            os.remove(msa_dir_path)
            logging.info(f"Temporary MSA directory {msa_dir_path} removed")
    else:
        logging.warning("No YAML file uploaded")
        st.error("Please upload a YAML file.")
