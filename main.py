# modal serve .\main.py
from typing import Annotated

import modal
from fastapi import UploadFile, Form, FastAPI, File, Header, HTTPException, status
import tempfile
from starlette.responses import JSONResponse

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "llama-index>=0.12.23",
    "llama-index-readers-file>=0.4.6",
    "unstructured[all-docs]>=0.16.25",
    "modal>=0.73.91",
    "fastapi[standard]==0.115.11",
    "pydantic==2.9.2",
    "python-multipart>=0.0.20",
    "starlette==0.41.2",
).apt_install(
    "libmagic1",
    "poppler-utils",
    "tesseract-ocr",
    "tesseract-ocr-eng",
    "libreoffice",
    "pandoc",
    "libmagic-dev",
)

app = modal.App(
    name="agentset-ingest",
    image=image,
    secrets=[
        modal.Secret.from_name("sec-ingest-1")
    ],
)

web_app = FastAPI()

@web_app.post("/ingest")
async def ingest(
        file: UploadFile = File(...),
        unstructured_args: Annotated[dict, Form()] = {
              "strategy": "auto",  # fast, hi_res, auto
            "chunking_strategy": "basic",  # by_title, basic
        },
        api_key: Annotated[str | None, Header()] = None
):
    import os

    if api_key != os.getenv('AGENTSET_API_KEY'):
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="api-key is not valid!",
            headers={"api-key": "<API_KEY>"},
        )

    from pathlib import Path
    from llama_index.readers.file import UnstructuredReader


    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        documents = UnstructuredReader().load_data(
            file=Path(file_path),
            unstructured_kwargs=unstructured_args
        )

        if len(documents) <= 0:
            return JSONResponse({"status": 500, "message": "coudn't parse document"})

        print('Done!')

        return JSONResponse(content={"status": 200, "documents": documents})


@app.function(timeout=1200)
@modal.asgi_app()
def fastapi_app():
    return web_app