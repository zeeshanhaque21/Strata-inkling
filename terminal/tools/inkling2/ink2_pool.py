"""ink2_pool.py: one OpenAI endpoint in front of several ink2_shim.py backends (one strata-ink2 engine per GPU).

Each engine serves one sequence at a time.  The pool sends a request to an idle backend, or queues it until one is free, so N backends
give N sequences generating at once and any further clients wait their turn (first come, first served).  Streaming responses are
passed through chunk by chunk.  /v1/models and /health are answered from the first healthy backend.

usage: ink2_pool.py --backend http://127.0.0.1:8101 --backend http://127.0.0.1:8102 [--host 0.0.0.0] [--port 8080] [--max-queue 64]
"""
import argparse
import asyncio

import aiohttp
from aiohttp import web


class Pool:
    def __init__(self, backends, max_queue):
        self.backends = backends
        self.free = None
        self.max_queue = max_queue
        self.waiting = 0
        self.served = {b: 0 for b in backends}

    def start(self):
        self.free = asyncio.Queue()
        for b in self.backends:
            self.free.put_nowait(b)

    async def acquire(self):
        if self.free.empty() and self.waiting >= self.max_queue:
            return None
        self.waiting += 1
        try:
            return await self.free.get()
        finally:
            self.waiting -= 1

    def release(self, b):
        self.served[b] += 1
        self.free.put_nowait(b)


async def proxy(request: web.Request):
    pool: Pool = request.app["pool"]
    body = await request.read()
    b = await pool.acquire()
    if b is None:
        return web.json_response({"error": {"message": "queue full"}}, status=503)
    try:
        async with request.app["http"].post(b + request.path, data=body, headers={"Content-Type": "application/json"}) as r:
            resp = web.StreamResponse(status=r.status, headers={k: v for k, v in r.headers.items()
                                                                if k.lower() in ("content-type", "cache-control")})
            await resp.prepare(request)
            async for chunk in r.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    except (aiohttp.ClientError, ConnectionResetError, asyncio.CancelledError):
        # client went away or the backend broke: closing the upstream connection makes the shim send STOP to its engine
        raise
    finally:
        pool.release(b)


async def get_any(request: web.Request):
    pool: Pool = request.app["pool"]
    for b in pool.backends:
        try:
            async with request.app["http"].get(b + request.path, timeout=aiohttp.ClientTimeout(total=5)) as r:
                if request.path.startswith("/health"):
                    return web.json_response({"status": "ok", "backends": len(pool.backends), "idle": pool.free.qsize(),
                                              "queued": pool.waiting, "served": list(pool.served.values())})
                return web.Response(body=await r.read(), status=r.status, content_type="application/json")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            continue
    return web.json_response({"error": {"message": "no backend"}}, status=503)


async def on_start(app):
    app["http"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10))
    app["pool"].start()


async def on_stop(app):
    await app["http"].close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", action="append", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-queue", type=int, default=64)
    a = ap.parse_args()
    app = web.Application()
    app["pool"] = Pool(a.backend, a.max_queue)
    app.router.add_post("/v1/chat/completions", proxy)
    app.router.add_get("/v1/models", get_any)
    app.router.add_get("/health", get_any)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    web.run_app(app, host=a.host, port=a.port, print=lambda *_: None)


if __name__ == "__main__":
    main()
