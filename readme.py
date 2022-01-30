import streamlit as st


def main():
    st.header("test")
    st.markdown("# test1")
    st.latex(r"\bar{X} = \frac{1}{N} \sum_{n=1}^{N} x_i")
    with st.expander("See details"):
        st.write("Hidden item")
        st.latex(r"\bar{X} = \frac{1}{N} \sum_{n=1}^{N} x_i")
        st.code("print('Hello, World!')")

    st.code("print('Hello, World!')")
    st.info("Please select the app")
    st.stop()


if __name__ == "__main__":
    main()
