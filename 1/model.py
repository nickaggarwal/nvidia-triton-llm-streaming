import json

import numpy as np
import triton_python_backend_utils as pb_utils
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer
from awq import AutoAWQForCausalLM

MODEL_NAME = "TheBloke/zephyr-7B-beta-AWQ"

class TritonPythonModel:
    """
    This model takes  input tensor, a STRING input named "TEST", and
    produces an output tensor "OUT" with the STRING Response
    if the request input has value 2, the model will:
        - Send a "Test" as input 
        - Keeps on sending steaming response
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

        # Load the Model 
        self.model = AutoAWQForCausalLM.from_quantized(MODEL_NAME, fuse_layers=False, version="GEMV")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)



    def execute(self, requests):
        for request in requests:
            in_input = pb_utils.get_input_tensor_by_name(request, "TEXT")

            string_data = str(in_input.as_numpy())
            print(string_data)

            messages = [{ "role": "system", "content": "You are an agent that know about about cooking." }] 

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
            response_sender = request.get_response_sender()

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            for new_text in self.streamer:
                print(new_text, flush=True)
                out_output = pb_utils.Tensor(
                    "OUT", np.array([new_text.encode('ascii')])
                )
                response = pb_utils.InferenceResponse(output_tensors=[out_output])
                response_sender.send(response)
            thread.join()

            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )

        return None
