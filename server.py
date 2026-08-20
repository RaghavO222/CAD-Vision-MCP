import os
import shutil
import json
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import your pipeline function from main.py
from main import auto_analyze_drawing, label_and_snapshot_dxf

app = FastAPI(title="Telecom CAD Analyzer")

# Ensure required directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("dxf_snap", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static and snapshot directories for frontend access
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/dxf_snap", StaticFiles(directory="dxf_snap"), name="dxf_snap")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.post("/api/analyze")
async def analyze_cad(
    dxf_file: UploadFile = File(...),
    ref_image: UploadFile = File(...),
    input_params: str = Form(...),
    azimuths: str = Form(...)  # Expected JSON string like "[0, 120, 240]" or comma-separated
):
    try:
        # 1. Save uploaded files
        dxf_path = os.path.join("uploads", dxf_file.filename)
        with open(dxf_path, "wb") as buffer:
            shutil.copyfileobj(dxf_file.file, buffer)

        img_path = os.path.join("uploads", ref_image.filename)
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(ref_image.file, buffer)

        # 2. Parse azimuths
        try:
            parsed_azimuths = json.loads(azimuths)
        except json.JSONDecodeError:
            parsed_azimuths = [a.strip() for a in azimuths.split(",") if a.strip()]

        # 3. Run analysis pipeline
        result = await auto_analyze_drawing(
            file=dxf_path,
            img=img_path,
            input_params=input_params,
            azimuths=parsed_azimuths
        )

        if result.get("status") != "success":
            return {"success": False, "error": result.get("reason", "Analysis failed")}

        # 4. Generate final snapshot of the modified DXF with proposed radiation
        final_snapshot = await label_and_snapshot_dxf(
            dxf_path=result["saved_file"],
            output_img_name="dxf_snap"
        )

        result["final_snapshot_path"] = final_snapshot.get("saved_snapshot")

        return {"success": True, "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)