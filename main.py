try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import db.init
from api.routes.brief import router as brief_router
from api.routes.calibration import router as calibration_router
from api.routes.collect import router as collect_router
from api.routes.health import router as health_router
from api.routes.history import router as history_router
from api.routes.output import router as output_router
from api.routes.pipeline import router as pipeline_router
from api.routes.signals import router as signals_router
from api.routes.summaries import router as summaries_router
from api.scheduler import start_scheduler, stop_scheduler
from delta.seeder import seed_baselines


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init.get_conn()
    seeded = seed_baselines()
    if seeded:
        print(f"[SigDriftr] Seeded {seeded} baseline rows.")
    start_scheduler()
    try:
        yield
    finally:
        stop_scheduler()


app = FastAPI(title="SigDriftr", lifespan=lifespan)


app.include_router(collect_router, prefix="")
app.include_router(signals_router, prefix="")
app.include_router(calibration_router, prefix="")
app.include_router(history_router, prefix="")
app.include_router(brief_router, prefix="")
app.include_router(health_router, prefix="")
app.include_router(pipeline_router, prefix="")
app.include_router(output_router, prefix="")
app.include_router(summaries_router, prefix="")
app.mount(
    "/ui",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="ui",
)
