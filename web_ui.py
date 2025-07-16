import os

import streamlit as st
from dotenv import load_dotenv
from kernel_opt.decorator.agent import KernelAgent
from kernel_opt.utils.proxy_util import devgpu_proxy_setup
from openai import OpenAI

FONT_SIZE = "15px"


def change_label_style(
    label, font_size="12px", font_color="black", font_family="sans-serif"
):
    html = f"""
    <script>
        var elems = window.parent.document.querySelectorAll('p');
        var elem = Array.from(elems).find(x => x.innerText == '{label}');
        elem.style.fontSize = '{font_size}';
        elem.style.color = '{font_color}';
        elem.style.fontFamily = '{font_family}';
    </script>
    """
    st.components.v1.html(html)


def main():
    # Set up the proxy for devgpu
    original_proxy_env = devgpu_proxy_setup()

    st.set_page_config(layout="wide")
    st.title("🔱  Triton Kernel Optimization Agent")

    st.caption("🚀 A Triton kernel performance optimization chatbot powered by LLMs")

    model_type = ""
    model = ""
    with st.sidebar:
        st.title("Model Settings")
        # LLM provider selection - expandable for new providers
        model_type = st.selectbox(
            "🏢 Select LLM Provider", ["OpenAI", "Google", "Anthropic", "Deepseek"]
        )
        if model_type == "OpenAI":
            model = st.selectbox(
                "🕹️ Select Model",
                [
                    "gpt-3.5-turbo",
                    "gpt-4",
                    "o3-2025-04-16",
                    "o4-mini-2025-04-16",
                ],
            )
        elif model_type == "Google":
            model = st.selectbox(
                "Select Model",
                [
                    "gemini-2.0-flash",
                ],
            )
        elif model_type == "Anthropic":
            model = st.selectbox(
                "Select Model",
                [
                    "claude-opus-4-20250514",
                ],
            )
        elif model_type == "Deepseek":
            model = st.selectbox(
                "Select Model",
                [
                    "deepseek-chat",
                ],
            )
        # API key management
        user_api_key = st.text_input("🔑 Your API Key", type="password")

        st.divider()
        openai_api_key = st.text_input(
            "🔑 Your OpenAI API Key (**required**)", type="password"
        )

        st.divider()
        debug = st.checkbox("🐞 Debug Mode", value=False)

        st.divider()
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[Get a Google API key](https://aistudio.google.com/app/u/2/apikey)"
        "[Get an Anthropic API key](https://docs.anthropic.com/en/api/admin-api/apikeys/get-api-key)"
        "[Get a DeepSeek API key](https://platform.deepseek.com/api_keys)"

    col1, col2 = st.columns(2)
    with col1:
        dsl = st.selectbox(
            r"$\textsf{\normalsize 📜 Language (only Triton supported now)}$",
            [
                "Triton",
                "Gluon",
            ],
        )
    with col2:
        kernel_name = st.text_input(
            r"$\textsf{\normalsize 📍 Kernel Name}$", placeholder="Kernel name?"
        )

    # Kernel function input
    kernel_function = st.text_input(
        r"$\textsf{\normalsize {📠}{~{Kernel Function}}}$",
        placeholder="A brief description of your kernel function...",
    )
    # Program input
    program_input = st.text_area(
        r"$\textsf{\normalsize 📝 Input Program}$",
        placeholder="What program do you want to optimize?",
        height=200,
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hello! I'm an agent 🤖 here to help you optimize your Triton kernel.\
                \n Could you tell me how you would like to optimize your kernel?",
            }
        ]

    if "program_input" not in st.session_state:
        st.session_state["program_input"] = program_input

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if opt_prompt := st.chat_input():
        if not user_api_key:
            st.info("Please add your API key to continue.")
            st.stop()
        if not model:
            st.info("Please select a model to continue.")
            st.stop()

        # Load LLM API Keys
        if model_type == "OpenAI":
            os.environ["OPENAI_API_KEY"] = user_api_key
        elif model_type == "Google":
            os.environ["GOOGLE_API_KEY"] = user_api_key
        elif model_type == "Anthropic":
            os.environ["ANTHROPIC_API_KEY"] = user_api_key
        elif model_type == "Deepseek":
            os.environ["DEEPSEEK_API_KEY"] = user_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # client = OpenAI(api_key=user_api_key)
        st.session_state.messages.append({"role": "user", "content": opt_prompt})
        st.chat_message("user").write(opt_prompt)

        with st.spinner("Optimizing your kernel...", show_time=True):
            status_msg, func_out, debug_msg, session_info = KernelAgent(
                model,
                dsl,
                kernel_name,
                kernel_function,
                st.session_state.program_input,
                opt_prompt,
                debug,
            )
        st.success("Done!")
        st.button("Rerun")

        st.session_state.messages.append({"role": "assistant", "content": func_out})
        with st.chat_message("assistant"):
            st.code(func_out, language="python", line_numbers=True, wrap_lines=False)

        if debug:
            session_expander = st.expander("Session Info")
            session_expander.text(session_info)
            debug_expander = st.expander("Debug Info")
            debug_expander.text(debug_msg)
            program_input_expander = st.expander("Program Input")
            program_input_expander.code(
                st.session_state.program_input,
                language="python",
                line_numbers=True,
                wrap_lines=False,
            )

        st.session_state.program_input = func_out


if __name__ == "__main__":
    main()
