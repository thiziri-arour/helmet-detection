# Adding the helmet model to the Vision Dashboard

Goal: choose "ppe" in the dashboard model menu, upload a construction video, see the workers tracked with helmet / no_helmet and get a saved report ("2 workers without a helmet").

## 1. Copy the files

- `weights/ppe_best.pt` -> `backend/models/ppe_best.pt`
- `integration/ppe_report.py` -> `backend/ppe_report.py`

## 2. `detector.py`: add the model to the list

Add the model next to the others:

```python
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

AVAILABLE_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8n-seg": "yolov8n-seg.pt",
    "ppe": str(MODELS_DIR / "ppe_best.pt"),   # new
}
```

and make sure the loader uses this dict:

```python
def get_model(name):
    if name not in _cache:
        _cache[name] = YOLO(AVAILABLE_MODELS[name])
    return _cache[name]
```

predict() and track() don't change, the class names are saved in the .pt file.

## 3. `db.py`: create the report table

At startup, where the other tables are created:

```python
from ppe_report import create_table

create_table(conn)
```

## 4. `main.py`: build the report after tracking a video

In the video tracking endpoint, when the ppe model is used:

```python
from ppe_report import collect_tracks, summarize_tracks, save_report, list_reports

results = list(model.track(video_path, tracker="bytetrack.yaml", stream=True, persist=True))
# ... your existing code that draws frames and saves detections ...

ppe = None
if model_name == "ppe":
    tracks = collect_tracks(results, model.names)
    ppe = summarize_tracks(tracks)
    save_report(conn, file.filename, ppe)

return {..., "ppe": ppe}
```

And a small endpoint for the history:

```python
@app.get("/ppe/reports")
def ppe_reports():
    return list_reports(conn)
```

If the loop already goes frame by frame, no need for list(...), the tracks dict can be filled directly in the loop.

## 5. `frontend/app.py`: show it in Streamlit

Add the model to the menu:

```python
model_name = st.selectbox("Model", ["yolov8n", "yolov8n-seg", "ppe"])
```

After the response comes back:

```python
ppe = data.get("ppe")
if ppe:
    col1, col2, col3 = st.columns(3)
    col1.metric("Workers with helmet", ppe["helmet"])
    col2.metric("Without helmet", ppe["no_helmet"])
    col3.metric("With vest", ppe["vest"])

    if ppe["no_helmet"] > 0:
        st.error(f"{ppe['no_helmet']} worker(s) without a helmet detected")
    else:
        st.success("Everyone is wearing a helmet")
```

And a history table:

```python
st.subheader("Safety reports")
reports = requests.get(f"{API_URL}/ppe/reports").json()
st.dataframe(reports)
```

## 6. Test it

Start backend + frontend, pick ppe and upload a short construction video (Pexels has free ones). Check the boxes, the 3 metrics and that a new row shows up in the reports table.
