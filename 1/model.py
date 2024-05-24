# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json

import numpy as np
import triton_python_backend_utils as pb_utils
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer
from awq import AutoAWQForCausalLM

MODEL_NAME = "TheBloke/zephyr-7B-beta-AWQ"


class TritonPythonModel:
    """
    This model takes 1 input tensor, an INT32 [ 1 ] input named "IN", and
    produces an output tensor "OUT" with the same shape as the input tensor.
    The input value indicates the total number of responses to be generated and
    the output value indicates the number of remaining responses. For example,
    if the request input has value 2, the model will:
        - Send a response with value 1.
        - Release request with RESCHEDULE flag.
        - When execute on the same request, send the last response with value 0.
        - Release request with ALL flag.
    """

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(
                    args["model_name"]
                )
            )

        # Get IN configuration

        self.remaining_response = 0
        self.reset_flag = True
        self.model = AutoAWQForCausalLM.from_quantized(MODEL_NAME, fuse_layers=False, version="GEMV")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)



    def execute(self, requests):
        for request in requests:
            in_input = pb_utils.get_input_tensor_by_name(request, "TEXT")
            import base64

            string_data = str(in_input.as_numpy())
            print(string_data)

            messages = [{ "role": "system", "content": "You are an agent that know about about cooking." }] 
            output=base64.b64encode(b'GeeksForGeeks')

            messages.append({ "role": "user", "content": string_data })
            tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()

            generation_kwargs = dict(
                inputs=tokenized_chat,
                streamer=self.streamer,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.2,
                max_new_tokens=1024,
            )

            out_output = pb_utils.Tensor(
                "OUT", np.array([output])
            )
            response = pb_utils.InferenceResponse(output_tensors=[out_output])
            response_sender = request.get_response_sender()

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            for new_text in self.streamer:
                print(new_text, flush=True)
                out_output = pb_utils.Tensor(
                    "OUT", np.array([new_text.encode('utf-8')])
                )
                response = pb_utils.InferenceResponse(output_tensors=[out_output])
                response_sender.send(response)
            thread.join()

            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )

        return None