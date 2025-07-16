"""Kernel generator."""

import os
from ast import Tuple
from pathlib import Path
from typing import Optional, Tuple

from kernel_opt.configs.models import (
    check_model_is_available,
    get_available_model_names,
    get_model,
)
from kernel_opt.utils.proxy_util import get_meta_proxy_config
from openai import OpenAI


class KernelRewriter:
    """Rewriter for the input kernel."""

    def __init__(
        self,
        prompt: str,
        model: str,
        debug: bool,
        module_path: Path,
        error: Optional[str] = None,
    ):
        """Initialize the rewriter.
        :param prompt: Description of the kernel to generate
        :param model: LLM model to use
        :param debug: Whether to print debug information
        """

        self.prompt = prompt
        self.model = model
        self.module_path = module_path
        self.debug = debug

        # Configure the proxy
        self._original_proxy_env = {}
        proxy_config = get_meta_proxy_config()
        if proxy_config:
            for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
                self._original_proxy_env[key] = os.environ.get(key)
                proxy_url = proxy_config.get(key)
                if proxy_url:
                    os.environ[key] = proxy_url

        self._model_object = get_model(model)
        self.model_name = self._model_object.name
        self.provider = self._model_object.provider()
        # self.llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate_kernel(self, error: Optional[str] = None) -> Tuple[str, str]:
        """Generate the kernel.
        :param error: Error message to include in the prompt
        """
        # Generate the kernel
        prompt = self.prompt
        if error:
            prompt = f"{error} \n" + prompt

        response_output = self.provider.get_response(
            model_name=self.model_name, prompt=prompt
        )
        # response = self.llm_client.chat.completions.create(
        #     model = self.model,
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": prompt}
        #     ],
        #     max_tokens=4096,
        # )
        # response_output = response.choices[0].message.content

        debug_str = ""
        if self.debug:
            debug_str += f"""
****** Response ****** :
{response_output}
"""
            # if str(self.module_path) != "":
            #     debug_response_path = self.module_path / "debug_output" / "response.log"
            #     with open(str(debug_response_path), "w") as file:
            #         file.write("****** Response ****** : \n")
            #         file.write(response_output)

        return response_output, debug_str
