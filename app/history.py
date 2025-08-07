from fastapi.staticfiles import StaticFiles
from collections import Counter
from datetime import datetime

@app.get("/history", response_class=HTMLResponse)
async def view_history(request: Request):
    if not os.path.exists(LOG_FILE):
        return templates.TemplateResponse("report.html", {
            "request": request,
            "summary": {},
            "logs": [],
            "chart_data": {"dates": [], "counts": []}
        })

    # Load log
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    # Grup berdasarkan tanggal
    grouped = {}
    for log in logs:
        date = log["timestamp"].split("T")[0]
        prediction = log.get("prediction", "Unknown")
        if date not in grouped:
            grouped[date] = {}
        grouped[date][prediction] = grouped[date].get(prediction, 0) + 1

    # Statistik untuk grafik
    dates = sorted(grouped.keys())
    total_counts = [sum(grouped[d].values()) for d in dates]

    # Statistik ringkasan
    attack_types = [log.get("prediction", "Unknown") for log in logs]
    most_common = Counter(attack_types).most_common(1)[0][0] if attack_types else "N/A"
    summary = {
        "days_analyzed": len(dates),
        "total_attacks": len([log for log in logs if "xss" in log.get("prediction", "").lower()]),
        "most_common_type": most_common
    }

    # Data tabel
    table_data = []
    for date in dates:
        for attack_type, count in grouped[date].items():
            table_data.append({
                "date": date,
                "attack_type": attack_type,
                "count": count
            })

    return templates.TemplateResponse("report.html", {
        "request": request,
        "summary": summary,
        "logs": table_data,
        "chart_data": {"dates": dates, "counts": total_counts}
    })
