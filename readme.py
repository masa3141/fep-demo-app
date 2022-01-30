import streamlit as st


def main():
    st.header("FEP Demo App")
    st.write(
        """
    This is an interactive demo application to understand the free energy principle(FEP) and active inference by using [streamlit](https://streamlit.io/) and [pymdp](https://github.com/infer-actively/pymdp).\n
    The contents are based on [pymdb tutorials](https://pymdp-rtd.readthedocs.io/en/latest/). \n
    Please select a task in the left sidebar.
    """
    )
    # st.latex(r"\bar{X} = \frac{1}{N} \sum_{n=1}^{N} x_i")
    # with st.expander("See details"):
    #     st.write("Hidden item")
    #     st.latex(r"\bar{X} = \frac{1}{N} \sum_{n=1}^{N} x_i")
    #     st.code("print('Hello, World!')")

    # st.code("print('Hello, World!')")
    # st.info("Please select the app")
    st.stop()


if __name__ == "__main__":
    main()
