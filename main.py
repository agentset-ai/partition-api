# modal serve .\main.py
from typing import Annotated
import json
from fastapi import UploadFile, Form, FastAPI, File, Header, status
import tempfile
from starlette.responses import JSONResponse

import modal
from llama_index.readers.file import UnstructuredReader
import requests

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "llama-index>=0.12.23",
        "llama-index-readers-file>=0.4.6",
        "unstructured[all-docs]>=0.16.25",
        "modal>=0.73.91",
        "fastapi[standard]==0.115.11",
        "pydantic==2.9.2",
        "python-multipart>=0.0.20",
        "starlette==0.41.2",
        "requests>=2.31.0",
    )
    .apt_install(
        "libmagic1",
        "poppler-utils",
        "tesseract-ocr",
        "tesseract-ocr-eng",
        "libreoffice",
        "pandoc",
        "libmagic-dev",
    )
)

app = modal.App(
    name="agentset-ingest",
    image=image,
    secrets=[modal.Secret.from_name("partitioner-secrets")],
)

web_app = FastAPI()


@web_app.post("/ingest")
async def ingest(
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
    filename: Annotated[str | None, Form()] = None,
    unstructured_args: Annotated[dict, Form()] = {
        "strategy": "auto",  # fast, hi_res, auto
        "chunking_strategy": "basic",  # by_title, basic
    },
    extra_metadata: Annotated[str | None, Form()] = None,
    api_key: Annotated[str | None, Header(alias="api-key")] = None,
):
    import os

    # Parse extra_metadata if provided
    metadata = None
    if extra_metadata:
        try:
            metadata = json.loads(extra_metadata)
            if not isinstance(metadata, dict):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "status": status.HTTP_400_BAD_REQUEST,
                        "message": "extra_metadata must be a valid JSON object",
                    },
                )
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "extra_metadata must be a valid JSON string",
                },
            )

    if api_key != os.getenv("AGENTSET_API_KEY"):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "status": status.HTTP_401_UNAUTHORIZED,
                "message": "api-key is not valid!",
            },
        )

    if not file and not url:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": status.HTTP_400_BAD_REQUEST,
                "message": "Either file or url must be provided",
            },
        )

    if file and url:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": status.HTTP_400_BAD_REQUEST,
                "message": "Cannot provide both file and url",
            },
        )

    if url and not filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": status.HTTP_400_BAD_REQUEST,
                "message": "filename is required when providing a URL",
            },
        )

    from pathlib import Path
    from llama_index.readers.file import UnstructuredReader
    import requests


    file_stream = None

    with tempfile.TemporaryDirectory() as temp_dir:
        if file:
            # file_path = os.path.join(temp_dir, file.filename)
            # with open(file_path, "wb") as f:
            #     f.write(await file.read())
            file_stream = await file.()
        else:
            try:
                response = requests.get(url)
                response.raise_for_status()
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "status": status.HTTP_400_BAD_REQUEST,
                        "message": f"Failed to download file from URL: {str(e)}",
                    },
                )

        try:
            documents = UnstructuredReader().load_data(
                unstructured_kwargs={
                    "file": Path(file_path),
                    "metadata_filename": filename,
                    **unstructured_args,
                },
                split_documents=True,
                extra_info=metadata,
            )

            if len(documents) <= 0:
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                        "message": "couldn't parse document",
                    },
                )

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": status.HTTP_200_OK,
                    "documents": [document.to_dict() for document in documents],
                },
            )
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": str(e),
                },
            )


@app.function(timeout=600)  # 10 minutes
@modal.asgi_app()
def partition_api():
    return web_app
