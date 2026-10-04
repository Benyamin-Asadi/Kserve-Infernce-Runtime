import argparse
import json
import os
import time
from typing import Dict

import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig

import kserve
from kserve import Model, ModelServer, logging

DS_CONFIG_PATH = os.environ.get("DS_CONFIG_PATH", "/app/ds_config.json")
MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/models")


class QwenZeroInferenceModel(Model):
    def __init__(self, name: str):
        super().__init__(name, return_response_headers=True)
        self.name = name
        self.tokenizer = None
        self.ds_engine = None
        self.dschf = None  # must be kept alive for the process lifetime — see load()
        self.ready = False
        self.load()

    def load(self):
        with open(DS_CONFIG_PATH) as f:
            ds_config = json.load(f)

        # Must be created BEFORE from_pretrained so HF shards/streams the
        # weight load according to the ZeRO config instead of loading the
        # full model into host RAM first and only then partitioning it.
        #
        # Stored on self deliberately, not as a local variable.
        # HuggingFace's own HfDeepSpeedConfig docs state this explicitly:
        # "A weakref of this object is stored in the module's globals...
        # it's important that this object remains alive while the program
        # is still running." A local variable here would go out of scope
        # (and could be garbage collected) as soon as load() returns,
        # which is exactly wrong for a KServe predictor where load() runs
        # once at startup and the process then serves indefinitely.
        self.dschf = HfDeepSpeedConfig(ds_config)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16,
        )
        model.eval()

        self.ds_engine = deepspeed.initialize(
            model=model,
            config_params=ds_config,
        )[0]
        self.ds_engine.module.eval()

        print("==== ZERO INFERENCE MODEL READY ====", flush=True)
        self.ready = True

    async def predict(
        self,
        payload: Dict,
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None,
    ) -> Dict:
        start = time.time()

        # Accepts either {"instances": ["prompt text"]} or
        # {"instances": [{"prompt": "...", "max_new_tokens": 64}]}
        instance = payload["instances"][0]
        if isinstance(instance, dict):
            prompt = instance["prompt"]
            max_new_tokens = instance.get("max_new_tokens", 64)
        else:
            prompt = instance
            max_new_tokens = 64

        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.ds_engine.module.device
        )

        with torch.no_grad():
            output_ids = self.ds_engine.module.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        end = time.time()
        if response_headers is not None:
            response_headers.update(
                {"prediction-time-latency": f"{round((end - start) * 1000, 3)}"}
            )

        return {"predictions": [generated]}


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    if args.configure_logging:
        logging.configure_logging(args.log_config_file)
    model = QwenZeroInferenceModel(args.model_name)
    ModelServer().start([model])
