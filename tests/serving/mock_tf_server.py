import numpy as np
from sanic import Sanic
from sanic.log import logger
from sanic.request import Request
from sanic.response import HTTPResponse, json

app = Sanic("mock-tf-server")


@app.post("/v1/models/<model_name:str>")
async def tf_serving(request: Request, model_name: str) -> HTTPResponse:
    logger.info(f"Mock predictions for {model_name.replace(':predict', '')}")
    rng = np.random.default_rng(42)
    payload = request.json["inputs"]
    if "k" in payload:
        return json({"outputs": rng.integers(0, 20, size=payload["k"]).tolist()})
    n_items = len(request.json["inputs"]["item_indices"])
    return json({"outputs": rng.normal(size=n_items).tolist()})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0", port=8501, debug=False, access_log=False, single_process=True
    )
