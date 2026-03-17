from fastapi import FastAPI

import db.init
from api.routes.collect import router as collect_router


app = FastAPI(title="SigDriftr")


@app.on_event("startup")
def startup() -> None:
    db.init.get_conn()


app.include_router(collect_router, prefix="")
