from fastapi import FastAPI

import db.init
from api.routes.calibration import router as calibration_router
from api.routes.collect import router as collect_router
from api.routes.signals import router as signals_router
from delta.seeder import seed_baselines


app = FastAPI(title="SigDriftr")


@app.on_event("startup")
def startup() -> None:
    db.init.get_conn()
    seeded = seed_baselines()
    if seeded:
        print(f"[SigDriftr] Seeded {seeded} baseline rows.")


app.include_router(collect_router, prefix="")
app.include_router(signals_router, prefix="")
app.include_router(calibration_router, prefix="")
