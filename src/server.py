"""
Tiny read-only HTTP view of the queue, so the report can be read from a phone.

Deliberately stdlib only: no FastAPI, no uvicorn, no template engine. It
serves plain text for four routes and nothing else. Adding a web framework for
a page with no forms and no writes would be weight without benefit.

Routes:
    /            the weekly report
    /clusters    every cluster and its members (to audit merging)
    /prompt/N    the cluster prompt for candidate N, ready to copy
    /healthz     for Railway's health check (no auth)

Everything except /healthz sits behind HTTP basic auth. Set AUTH_USER and
AUTH_PASS; if they are unset the server refuses to start rather than exposing
the data publicly.
"""
from __future__ import annotations

import base64
import logging
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import db
from .report.prompts import build_cluster_prompt
from .report.render import (
    find_alerts,
    group_into_clusters,
    load_queue,
    render_clusters_table,
    render_text,
)

log = logging.getLogger(__name__)


def _credentials() -> tuple[str, str]:
    user = os.environ.get("AUTH_USER", "")
    password = os.environ.get("AUTH_PASS", "")
    if not user or not password:
        raise SystemExit(
            "AUTH_USER and AUTH_PASS must be set. Refusing to serve the queue "
            "without authentication."
        )
    return user, password


class Handler(BaseHTTPRequestHandler):
    server_version = "aso-scout"

    def log_message(self, fmt: str, *args: object) -> None:
        log.info("%s %s", self.address_string(), fmt % args)

    # --- auth ------------------------------------------------------------
    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:]).decode("utf-8")
            user, _, password = decoded.partition(":")
        except Exception:
            return False
        want_user, want_pass = _credentials()
        # compare_digest on both parts: avoids leaking which one was wrong
        return (
            secrets.compare_digest(user, want_user)
            and secrets.compare_digest(password, want_pass)
        )

    def _challenge(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="aso-scout"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"auth required\n")

    def _text(self, body: str, status: int = 200) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        # text/plain with a viewport-friendly charset; the report is
        # monospace-formatted and reads fine on a phone.
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # --- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"

        if path == "/healthz":
            self._text("ok\n")
            return

        if not self._authorized():
            self._challenge()
            return

        try:
            if path == "/":
                self._text(self._report())
            elif path == "/clusters":
                self._text(self._clusters())
            elif path == "/digest":
                self._text(self._send_digest())
            elif path == "/export.csv":
                self._csv(self._export_csv(), "aso-scout-queue.csv")
            elif path == "/export.db":
                self._download_db()
            elif path.startswith("/prompt/"):
                self._text(self._prompt(path.rsplit("/", 1)[-1]))
            else:
                self._text(
                    "not found\n\n"
                    "/            отчёт\n"
                    "/clusters    все ниши таблицей\n"
                    "/prompt/N    промпт кластера N\n"
                    "/digest      отправить дайджест в Telegram сейчас\n"
                    "/export.csv  очередь файлом\n"
                    "/export.db   копия базы целиком\n",
                    404,
                )
        except Exception as exc:  # keep the server alive on a bad page
            log.exception("error serving %s", path)
            self._text(f"error: {exc}\n", 500)

    def _csv(self, body: str, filename: str) -> None:
        payload = body.encode("utf-8-sig")  # BOM so Excel reads Cyrillic
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _download_db(self) -> None:
        """
        Stream a consistent copy of the database.

        The volume is a single point of failure holding the one asset that
        cannot be rebuilt: snapshot history. sqlite3's backup API copies a
        live database safely, unlike reading the file while it is being
        written.
        """
        import sqlite3 as _sq  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        from pathlib import Path as _Path  # noqa: PLC0415

        from .config import settings as _settings  # noqa: PLC0415

        tmp = _Path(tempfile.mkdtemp()) / "aso_scout_backup.db"
        src = _sq.connect(_settings.sqlite_path)
        dst = _sq.connect(str(tmp))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()

        payload = tmp.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition",
                         'attachment; filename="aso_scout.db"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # --- page builders ---------------------------------------------------
    def _report(self) -> str:
        from .jobs.report import history_weeks  # noqa: PLC0415

        with db.connect() as conn:
            queue = load_queue(conn)
            return render_text(queue, find_alerts(conn), history_weeks(conn))

    def _clusters(self) -> str:
        with db.connect() as conn:
            return render_clusters_table(load_queue(conn), conn)

    def _send_digest(self) -> str:
        from .jobs.digest import build  # noqa: PLC0415
        from .notify.telegram import configured, send_digest  # noqa: PLC0415

        if not configured():
            return "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID не заданы\n"
        text, size = build()
        if not size:
            return "очередь пуста\n"
        ok = send_digest(text)
        return (f"отправлено, {size} запросов в очереди\n" if ok
                else "не удалось отправить — смотрите логи\n")

    def _export_csv(self) -> str:
        import csv  # noqa: PLC0415
        import io  # noqa: PLC0415

        from .report.render import group_into_clusters  # noqa: PLC0415

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["звёзды", "магазин", "ниша", "запросов", "score",
                    "лидер", "рейтинг", "отзывы", "устарело", "флаги",
                    "варианты"])
        with db.connect() as conn:
            for cl in group_into_clusters(load_queue(conn)):
                c, r = cl.head, cl.rating
                lead = c.leader or {}
                stale = c.features.get("stale_share")
                w.writerow([
                    r.stars, c.storefront.upper(), c.term, cl.size, c.score,
                    lead.get("title", ""), lead.get("rating", ""),
                    lead.get("rating_count", ""),
                    f"{int(stale * 10)}/10" if stale is not None else "",
                    " ".join(r.flags),
                    ", ".join(v.term for v in cl.variants),
                ])
        return buf.getvalue()

    def _prompt(self, raw: str) -> str:
        from .jobs.report import metadata_tokens, suggestions_for  # noqa: PLC0415

        try:
            n = int(raw)
        except ValueError:
            return "usage: /prompt/1\n"

        with db.connect() as conn:
            groups = group_into_clusters(load_queue(conn))
            if not 1 <= n <= len(groups):
                return f"нет ниши {n} (всего {len(groups)})\n"
            c = groups[n - 1].head
            return build_cluster_prompt(
                term=c.term,
                storefront=c.storefront,
                apps=c.apps,
                suggestions=suggestions_for(conn, c.term, c.storefront),
                metadata_tokens=metadata_tokens(c.apps),
            )


def serve(port: int | None = None) -> ThreadingHTTPServer:
    """Start the server. Validates credentials before binding."""
    _credentials()
    bind = int(port or os.environ.get("PORT", 8080))
    httpd = ThreadingHTTPServer(("0.0.0.0", bind), Handler)
    log.info("serving report on :%d", bind)
    return httpd


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    db.init_db()
    serve().serve_forever()
