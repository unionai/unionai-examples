"""A single-script Streamlit app example."""

import pathlib
import streamlit as st
import flyte
import flyte.app

# {{docs-fragment streamlit-app}}
st.set_page_config(page_title="Simple Streamlit App", page_icon="🚀")

st.title("Hello from Streamlit!")
st.write("This is a simple single-script Streamlit app.")

name = st.text_input("What's your name?", "World")
st.write(f"Hello, {name}!")

if st.button("Click me!"):
    st.balloons()
    st.success("Button clicked!")
# {{/docs-fragment streamlit-app}}

# {{docs-fragment app-env}}
app_env = flyte.app.AppEnvironment(
    name="streamlit-single-script",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "streamlit==1.41.1"
    ),
    command="streamlit run streamlit_single_script.py --server.port 8080",
    port=8080,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    requires_auth=False,
)
# {{/docs-fragment app-env}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.serve(app_env)
    print(f"App URL: {app[0].url}")
# {{/docs-fragment deploy}}

