"""Document byte storage: S3 in AWS, a local directory on a laptop.

Object keys are ``raw/{tenant_id}/{document_id}/v{version}``. They contain no
file names, so there is no URL-encoding ambiguity between the API, S3 event
notifications and the worker (the previous implementation needed a function
that tried four encodings of the same key).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from ragplatform.domain.errors import PayloadTooLarge, PermanentIngestError, UpstreamUnavailable
from ragplatform.domain.models import UploadInstruction

_KEY = re.compile(r"^raw/([a-z0-9][a-z0-9-]{1,62})/([0-9a-f-]{36})/v(\d{1,6})$")


def storage_key(tenant_id: str, document_id: UUID, version: int) -> str:
    return f"raw/{tenant_id}/{document_id}/v{version}"


def parse_storage_key(key: str) -> tuple[str, UUID, int]:
    match = _KEY.match(key)
    if not match:
        raise ValueError(f"not a document key: {key!r}")
    return match.group(1), UUID(match.group(2)), int(match.group(3))


class BlobStore(Protocol):
    def upload_instruction(
        self, key: str, *, size_bytes: int, content_type: str, local_url: str
    ) -> UploadInstruction: ...

    def put(self, key: str, data: bytes, content_type: str) -> None: ...

    def get(self, key: str, *, max_bytes: int) -> bytes: ...

    def delete_all_versions(self, key: str) -> int: ...


class S3BlobStore:
    def __init__(self, *, bucket: str, region: str, url_ttl_s: int, client: Any = None) -> None:
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3",
                region_name=region,
                config=Config(signature_version="s3v4", retries={"mode": "standard"}),
            )
        self.bucket = bucket
        self.url_ttl_s = url_ttl_s
        self._s3 = client

    def upload_instruction(
        self, key: str, *, size_bytes: int, content_type: str, local_url: str
    ) -> UploadInstruction:
        # A presigned POST policy can pin the exact byte count, which a
        # presigned PUT cannot. The worker re-checks size and SHA-256 anyway.
        post = self._s3.generate_presigned_post(
            Bucket=self.bucket,
            Key=key,
            Fields={"Content-Type": content_type},
            Conditions=[
                ["content-length-range", size_bytes, size_bytes],
                {"Content-Type": content_type},
            ],
            ExpiresIn=self.url_ttl_s,
        )
        return UploadInstruction(
            method="POST",
            url=post["url"],
            fields=post["fields"],
            expires_in_s=self.url_ttl_s,
            max_bytes=size_bytes,
        )

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str, *, max_bytes: int) -> bytes:
        from botocore.exceptions import ClientError

        try:
            head = self._s3.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise PermanentIngestError(
                    "missing_object", "the uploaded file was not found"
                ) from exc
            raise UpstreamUnavailable(f"s3 head_object: {code}") from exc
        if int(head["ContentLength"]) > max_bytes:
            raise PermanentIngestError("too_large", "the uploaded file exceeds the size limit")
        body: bytes = self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return body

    def delete_all_versions(self, key: str) -> int:
        """Delete every version and delete marker, so the bytes are actually gone.

        With versioning on, a plain delete only adds a delete marker and the
        data stays recoverable, which does not satisfy a deletion request.
        """
        deleted = 0
        paginator = self._s3.get_paginator("list_object_versions")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=key):
            objects = [
                {"Key": item["Key"], "VersionId": item["VersionId"]}
                for item in page.get("Versions", []) + page.get("DeleteMarkers", [])
                if item["Key"] == key
            ]
            if objects:
                response = self._s3.delete_objects(
                    Bucket=self.bucket, Delete={"Objects": objects, "Quiet": True}
                )
                if response.get("Errors"):
                    raise UpstreamUnavailable(f"s3 delete_objects errors: {response['Errors'][:3]}")
                deleted += len(objects)
        return deleted


class LocalBlobStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        parse_storage_key(key)  # rejects anything that is not a document key
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("key escapes the blob directory")
        return path

    def upload_instruction(
        self, key: str, *, size_bytes: int, content_type: str, local_url: str
    ) -> UploadInstruction:
        return UploadInstruction(
            method="PUT",
            url=local_url,
            fields={"Content-Type": content_type},
            expires_in_s=0,
            max_bytes=size_bytes,
        )

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str, *, max_bytes: int) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise PermanentIngestError("missing_object", "the uploaded file was not found")
        if path.stat().st_size > max_bytes:
            raise PayloadTooLarge("file exceeds the size limit")
        return path.read_bytes()

    def delete_all_versions(self, key: str) -> int:
        path = self._path(key)
        if path.exists():
            path.unlink()
            return 1
        return 0
