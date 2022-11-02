from fastapi import FastAPI
from fastapi import File
from ray import serve
import ray
from ray import runtime_env
from ray.runtime_env import RuntimeEnv
from time import time
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

app = FastAPI()

# runtime_env = RuntimeEnv(
#             working_dir = "/Users/apple/code/mlops/ray/raw-ocr",
#             pip = "req.txt",
# )

@serve.deployment(
    route_prefix="/ocr",
    num_replicas=2,
    ray_actor_options={
        "num_cpus": 2,
        "runtime_env": {
            "working_dir": ".",
            "pip": "req.txt",
        },
    },
)
@serve.ingress(app)
class RawOCR:
    def __init__(self):
        self.model = ocr_predictor(
            det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
        )
        print("============== INIT ================")

    @app.post("/")
    def ocr_pdf(self, pdf: bytes = File()) -> dict:
        doc = DocumentFile.from_pdf(pdf)
        t1 = time()
        result = self.model(doc)
        t2 = time()
        return {"ocr_text": result.render(), "processing_time": t2 - t1}


ocr = RawOCR.bind()
