# modal serve .\main.py
from typing import Annotated
import json
from fastapi import UploadFile, Form, FastAPI, File, Header, status
from starlette.responses import JSONResponse
import modal

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
    api_key: Annotated[str | None, Header(alias="api-key")] = None,
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
    text: Annotated[str | None, Form()] = None,
    filename: Annotated[str | None, Form()] = None,
    extra_metadata: Annotated[str | None, Form()] = None,
    unstructured_args_json: Annotated[
        str | None, Form(alias="unstructured_args")
    ] = None,
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

    unstructured_args = {
        "strategy": "auto",  # fast, hi_res, auto
        "chunking_strategy": "basic",  # by_title, basic
    }
    if unstructured_args_json:
        try:
            unstructured_args = json.loads(unstructured_args_json)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "unstructured_args must be a valid JSON string",
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

    # Count how many input sources are provided
    input_sources = sum(1 for x in [file, url, text] if x is not None)

    if input_sources == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": status.HTTP_400_BAD_REQUEST,
                "message": "Either file, url, or text must be provided",
            },
        )

    if input_sources > 1:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": status.HTTP_400_BAD_REQUEST,
                "message": "Only one of file, url, or text can be provided",
            },
        )

    if (url or text) and not filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": status.HTTP_400_BAD_REQUEST,
                "message": "filename is required when providing a URL or text",
            },
        )

    from io import BytesIO
    from llama_index.readers.file import UnstructuredReader
    from unstructured.file_utils.filetype import detect_filetype
    import requests

    file_stream = None
    filename_to_use = file.filename if file else filename
    size_in_bytes = 0

    try:
        if file:
            size_in_bytes = file.size
            file_stream = BytesIO(await file.read())
        elif url:
            try:
                response = requests.get(url)
                response.raise_for_status()
                size_in_bytes = len(response.content)
                file_stream = BytesIO(response.content)
            except Exception as e:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "status": status.HTTP_400_BAD_REQUEST,
                        "message": f"Failed to download file from URL: {str(e)}",
                    },
                )
        else:  # text input
            text_bytes = text.encode("utf-8")
            size_in_bytes = len(text_bytes)
            file_stream = BytesIO(text_bytes)

        content_type = detect_filetype(
            file=file_stream,
            metadata_file_path=filename_to_use,
        ).mime_type
        documents = UnstructuredReader(
            allowed_metadata_types=(str, int, float, list, dict, type(None)),
        ).load_data(
            unstructured_kwargs={
                "file": file_stream,
                "metadata_filename": filename_to_use,
                "content_type": content_type,
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

        result_documents = []
        total_characters = 0
        total_chunks = 0
        for document in documents:
            total_chunks += 1
            result_documents.append(document.to_dict())
            if document.text:
                total_characters += len(document.text)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": status.HTTP_200_OK,
                "metadata": {
                    "filename": filename_to_use,
                    "filetype": content_type,
                    "sizeInBytes": size_in_bytes,
                },
                "total_characters": total_characters,
                "total_chunks": total_chunks,
                "chunks": result_documents,
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
