"""Streamlit client for the RAG platform API.

A thin client: every action is one API call, and nothing here decides what a
user may see. Identity, access control, retrieval and answer validation all
happen behind the API.

    streamlit run ui/app.py

Settings (environment variables):
    RAG_UI_API_URL            API base URL, e.g. http://localhost:8000 or the deployed stage URL
    RAG_UI_AUTH_MODE          local (paste or mint a development token) or cognito
    RAG_UI_COGNITO_CLIENT_ID  app client id (cognito mode)
    RAG_UI_REGION             AWS region of the user pool (cognito mode)
    RAG_LOCAL_AUTH_SECRET     local mode only: lets the sign-in form mint tokens
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import time
from typing import Any

import requests
import streamlit as st

API_URL = os.environ.get("RAG_UI_API_URL", "http://localhost:8000").rstrip("/")
AUTH_MODE = os.environ.get("RAG_UI_AUTH_MODE", "local")
TIMEOUT_S = 35

STATUS_LABELS = {
    "answered": "Answered from your documents",
    "insufficient_context": "Not enough information in the documents you can access",
    "conflicting_sources": "Sources disagree",
}

st.set_page_config(page_title="RAG Platform", layout="wide")

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


def safe_markdown(text: str) -> str:
    """Answer text with images removed and links shown as plain text.

    An answer can repeat text from a poisoned document. A markdown image would
    make the browser fetch its URL, which can carry data in the query string,
    so images are dropped and links are not made clickable.
    """
    text = _MD_IMAGE.sub("[image removed]", text)
    return _MD_LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text).replace("<", "&lt;")


# --------------------------------------------------------------------------- API helpers


def api(method: str, path: str, **kwargs: Any) -> requests.Response:
    headers = kwargs.pop("headers", {})
    token = st.session_state.get("token")
    if token:
        # API Gateway's Cognito authorizer reads the raw ID token; the local
        # server accepts the same header with a Bearer prefix.
        headers["Authorization"] = token if AUTH_MODE == "cognito" else f"Bearer {token}"
    return requests.request(
        method, f"{API_URL}{path}", headers=headers, timeout=TIMEOUT_S, **kwargs
    )


def show_error(response: requests.Response) -> None:
    try:
        error = response.json()["error"]
        message = f"{error['message']} (code {error['code']}, request {error['request_id']})"
    except (ValueError, KeyError):
        message = f"HTTP {response.status_code}"
    if response.status_code == 503:
        retry = response.headers.get("Retry-After")
        message += f". Retry in {retry} s." if retry else ""
    st.error(message)


# --------------------------------------------------------------------------- sign-in


def cognito_client() -> Any:
    import boto3

    return boto3.client("cognito-idp", region_name=os.environ.get("RAG_UI_REGION", "us-east-1"))


def cognito_sign_in() -> None:
    client_id = os.environ["RAG_UI_COGNITO_CLIENT_ID"]
    challenge = st.session_state.get("challenge")
    if challenge is None:
        with st.form("sign-in"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in"):
                try:
                    result = cognito_client().initiate_auth(
                        ClientId=client_id,
                        AuthFlow="USER_PASSWORD_AUTH",
                        AuthParameters={"USERNAME": email, "PASSWORD": password},
                    )
                except Exception as exc:  # noqa: BLE001 - shown to the user as a sign-in failure
                    st.error(f"Sign-in failed: {type(exc).__name__}")
                    return
                handle_auth_result(result, email)
        return

    name, session, email = challenge
    with st.form("challenge"):
        if name == "NEW_PASSWORD_REQUIRED":
            st.info("Set a new password (at least 14 characters).")
            answer = st.text_input("New password", type="password")
            key = "NEW_PASSWORD"
        else:  # SOFTWARE_TOKEN_MFA
            answer = st.text_input("Code from your authenticator app")
            key = "SOFTWARE_TOKEN_MFA_CODE"
        if st.form_submit_button("Continue"):
            try:
                result = cognito_client().respond_to_auth_challenge(
                    ClientId=client_id,
                    ChallengeName=name,
                    Session=session,
                    ChallengeResponses={"USERNAME": email, key: answer},
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not complete sign-in: {type(exc).__name__}")
                return
            handle_auth_result(result, email)


def handle_auth_result(result: dict[str, Any], email: str) -> None:
    if "AuthenticationResult" in result:
        st.session_state.token = result["AuthenticationResult"]["IdToken"]
        st.session_state.user = email
        st.session_state.pop("challenge", None)
        st.rerun()
    elif result.get("ChallengeName") in {"NEW_PASSWORD_REQUIRED", "SOFTWARE_TOKEN_MFA"}:
        st.session_state.challenge = (result["ChallengeName"], result["Session"], email)
        st.rerun()
    else:
        st.error(f"Unsupported sign-in step: {result.get('ChallengeName')}")


def local_sign_in() -> None:
    st.caption("Local development sign-in. Tokens are signed with RAG_LOCAL_AUTH_SECRET.")
    secret = os.environ.get("RAG_LOCAL_AUTH_SECRET")
    with st.form("local"):
        if secret:
            user = st.text_input("User id", "alice")
            tenant = st.text_input("Tenant", "acme")
            groups = st.text_input("Groups (comma-separated)", "staff")
        token = st.text_area("Or paste a token from `ragplatform token`", "")
        if st.form_submit_button("Sign in"):
            if token.strip():
                st.session_state.token, st.session_state.user = token.strip(), "token"
            elif secret:
                from ragplatform.security.auth import issue_local_token

                st.session_state.token = issue_local_token(
                    secret,
                    user_id=user,
                    tenant_id=tenant,
                    groups=[g.strip() for g in groups.split(",") if g.strip()],
                )
                st.session_state.user = f"{user}@{tenant}"
            st.rerun()


# --------------------------------------------------------------------------- pages


def upload_page() -> None:
    st.subheader("Add a document")
    file = st.file_uploader(
        "PDF, DOCX, HTML, Markdown, text or CSV",
        type=["pdf", "docx", "html", "htm", "md", "txt", "csv"],
    )
    acl = st.text_input("Share with groups (comma-separated, groups you belong to)", "staff")
    title = st.text_input("Title (optional; taken from the document when empty)", "")
    if file is None or not st.button("Upload"):
        return
    data = file.getvalue()
    content_type = file.type or mimetypes.guess_type(file.name)[0] or ""
    body = {
        "source_name": file.name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "content_type": content_type,
        "acl_groups": [g.strip() for g in acl.split(",") if g.strip()],
    }
    if title.strip():
        body["title"] = title.strip()
    registered = api("POST", "/v1/documents", json=body)
    if registered.status_code != 201:
        show_error(registered)
        return
    payload = registered.json()
    if payload["duplicate"]:
        st.info("You already uploaded this file. Nothing to do.")
        return
    upload = payload["upload"]
    if upload["method"] == "POST":  # S3 presigned POST
        response = requests.post(
            upload["url"], data=upload["fields"], files={"file": (file.name, data)}, timeout=120
        )
    else:  # local development server
        response = api(
            "PUT",
            upload["url"],
            data=data,
            headers={"Content-Type": upload["fields"].get("Content-Type", "")},
        )
    if response.status_code not in (200, 202, 204):
        st.error(f"Upload failed with HTTP {response.status_code}")
        return
    document_id = payload["document"]["document_id"]
    with st.status("Processing", expanded=False) as status:
        for _ in range(60):
            detail = api("GET", f"/v1/documents/{document_id}")
            state = detail.json()["document"]["status"] if detail.ok else "unknown"
            if state in {"ready", "failed"}:
                break
            time.sleep(2)
        if state == "ready":
            status.update(label="Ready to search", state="complete")
        else:
            versions = detail.json().get("versions", []) if detail.ok else []
            reason = versions[-1].get("error_detail") if versions else None
            status.update(
                label=f"Processing {state}: {reason or 'see the document list'}", state="error"
            )


def documents_page() -> None:
    st.subheader("Documents you can access")
    response = api("GET", "/v1/documents?limit=50")
    if not response.ok:
        show_error(response)
        return
    items = response.json()["items"]
    if not items:
        st.write("No documents yet.")
        return
    for item in items:
        cols = st.columns([5, 2, 2, 1])
        cols[0].write(f"**{item['title']}**  \n{item['source_name']}")
        cols[1].write(item["status"])
        cols[2].write(", ".join(item["acl_groups"]) or "owner only")
        if cols[3].button("Delete", key=f"del-{item['document_id']}"):
            deleted = api("DELETE", f"/v1/documents/{item['document_id']}")
            if deleted.ok:
                st.rerun()
            else:
                show_error(deleted)


def ask_page() -> None:
    st.subheader("Ask a question")
    question = st.text_area("Question", height=80)
    cols = st.columns(3)
    agentic = cols[0].toggle("Agentic mode", help="Plans and runs several searches. Slower.")
    debug = cols[1].toggle("Show retrieval ranks")
    if not st.button("Ask", type="primary") or not question.strip():
        return
    with st.spinner("Searching and answering"):
        response = api(
            "POST",
            "/v1/query",
            json={
                "question": question,
                "mode": "agentic" if agentic else "standard",
                "debug": debug,
            },
        )
    if not response.ok:
        show_error(response)
        return
    result = response.json()
    label = STATUS_LABELS.get(result["status"], result["status"])
    if result["status"] == "answered":
        st.success(label)
    else:
        st.warning(label)
    st.markdown(safe_markdown(result["answer"]))
    for conflict in result.get("conflicts", []):
        st.info(conflict)
    if result["citations"]:
        st.markdown("**Sources**")
        for citation in result["citations"]:
            page = f", page {citation['page_start']}" if citation.get("page_start") else ""
            with st.expander(
                f"[{citation['marker']}] {citation['title']} "
                f"({citation['section_path'] or 'document'}{page}, v{citation['version']})"
            ):
                st.write(citation["snippet"])
    meta = (
        f"{result['latency_ms']:.0f} ms, {result['usage']['input_tokens']} input and "
        f"{result['usage']['output_tokens']} output tokens, model {result['model_id']}"
    )
    if result.get("cost_usd") is not None:
        meta += f", estimated cost ${result['cost_usd']:.5f}"
    if result.get("cached"):
        meta += ", served from cache"
    st.caption(meta)
    if result.get("retrieved"):
        st.dataframe(result["retrieved"], use_container_width=True)
    if result.get("agent_trace"):
        st.markdown("**Agent steps**")
        st.dataframe(result["agent_trace"], use_container_width=True)


# --------------------------------------------------------------------------- layout


def main() -> None:
    st.title("RAG Platform")
    if "token" not in st.session_state:
        cognito_sign_in() if AUTH_MODE == "cognito" else local_sign_in()
        return
    with st.sidebar:
        st.write(f"Signed in as **{st.session_state.get('user', '')}**")
        usage = api("GET", "/v1/usage")
        if usage.ok:
            today = usage.json()["today"]
            st.caption(
                f"Today: {today['queries']} questions, "
                f"{today['input_tokens'] + today['output_tokens']} tokens"
            )
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()
    ask, docs, add = st.tabs(["Ask", "Documents", "Add document"])
    with ask:
        ask_page()
    with docs:
        documents_page()
    with add:
        upload_page()


main()
