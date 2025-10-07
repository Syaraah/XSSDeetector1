import os
import re
import logging
import json
import csv
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_payload

from flask import (
    Flask, 
    Response, 
    make_response, 
    render_template, 
    request, 
    send_file, 
    jsonify
)
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from collections import defaultdict
from io import StringIO
from datetime import datetime
from collections import Counter
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  

# ========== Konfigurasi ==========
MODEL_PATH = 'models/xss_lstm_model.keras'
TOKENIZER_PATH = 'models/tokenizer.pkl'
LOG_PATH = 'logs/detection_log.json'
MAX_LEN = 200
THRESHOLD = 0.7  
DUPLICATE_WINDOW = 5  

# ========== Logging Setup ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== App Init ==========
app = Flask(__name__)
app.jinja_env.filters['zip'] = zip
app.config['SECRET_KEY'] = 'secretkey'

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# SocketIO with CORS support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# # # ========== Session Code & WebSocket Auth ==========
# # VALID_SESSION_CODES = {
# #     # 'session_code': 'web_name',
# #     'ABC123': 'WebA',
# #     'DEF456': 'WebB',
# #     # Tambahkan kode lain sesuai kebutuhan
# # }

# # # Mapping sid ke session code dan web
# # client_sessions = {}

# @socketio.on('join')
# def handle_join(data):
#     """Client mengirim session code untuk validasi awal"""
#     session_code = data.get('session_code')
#     sid = request.sid
#     if not session_code or session_code not in VALID_SESSION_CODES:
#         logger.warning(f"Client {sid} attempted join with invalid session code: {session_code}")
#         emit('join_result', {"error": "Invalid session code"})
#         return False  # Menolak koneksi lebih lanjut
#     # Simpan mapping
#     client_sessions[sid] = {
#         'session_code': session_code,
#         'web_name': VALID_SESSION_CODES[session_code]
#     }
#     logger.info(f"Client {sid} joined with session code {session_code} ({VALID_SESSION_CODES[session_code]})")
#     emit('join_result', {"success": True, "web_name": VALID_SESSION_CODES[session_code]})

# @socketio.on('submit_payload')
# def handle_submit_payload(data):
#     """Handle real-time payload submission dengan validasi session code"""
#     sid = request.sid
#     # Cek apakah client sudah join dengan session code valid
#     if sid not in client_sessions:
#         emit('detection_result', {"error": "Unauthorized: Session code required. Please join first."})
#         return
#     try:
#         payload = data.get('payload', '')
#         if not payload:
#             emit('detection_result', {"error": "Please enter a payload to analyze"})
#             return
#         # Validate payload
#         is_valid, error_msg = validate_payload(payload)
#         if not is_valid:
#             emit('detection_result', {"error": error_msg})
#             return
#         result = detect_payload(payload)
#         if 'error' not in result:
#             if save_log(result):
#                 socketio.emit('new_detection', result, include_self=False)
#         emit('detection_result', result)
#     except Exception as e:
#         logger.error(f"Socket error: {e}")
#         emit('detection_result', {"error": "Internal server error occurred"})

# @socketio.on('connect')
# def handle_connect():
#     logger.info(f"Client connected: {request.sid}")
#     emit('connected', {"message": "Connected to XSS Detection System"})

# @socketio.on('disconnect')
# def handle_disconnect():
#     sid = request.sid
#     if sid in client_sessions:
#         logger.info(f"Client disconnected: {sid} (session {client_sessions[sid]['session_code']})")
#         del client_sessions[sid]
#     else:
#         logger.info(f"Client disconnected: {sid}")

# ========== Load model & tokenizer ==========
print("Loading model and tokenizer...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    print("Model & tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

# ========== Helper Functions ==========
def validate_payload(payload):
    """Validate input payload"""
    if not payload or not isinstance(payload, str):
        return False, "Payload must be a non-empty string"
    
    if len(payload) > 10000:  
        return False, "Payload too long (max 10000 characters)"
    
    # Remove strict XSS pattern validation - let the model decide
    return True, "Valid payload"

def is_duplicate_payload(payload, timestamp):
    """Check if payload is duplicate within time window"""
    try:
        if not os.path.exists(LOG_PATH):
            return False
        
        with open(LOG_PATH, 'r') as f:
            logs = json.load(f)
        
        if not logs:
            return False
        
        # Check last 10 entries for duplicates within time window
        recent_logs = logs[-10:]
        current_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        for log in recent_logs:
            if log.get('payload') == payload:
                log_time = datetime.strptime(log.get('timestamp'), "%Y-%m-%d %H:%M:%S")
                time_diff = (current_time - log_time).total_seconds()
                if time_diff < DUPLICATE_WINDOW:
                    return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking duplicate: {e}")
        return False

def detect_payload(payload):
    try:
        if model is None or tokenizer is None:
            return {
                "error": "Model not loaded",
                "original": payload,
                "cleaned": clean_payload(payload),
                "label": "Error",
                "confidence": 0.0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        cleaned = clean_payload(payload)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        prob = float(model.predict(padded, verbose=0)[0][0])
        label = "Malicious" if prob > THRESHOLD else "Benign"
        
        # Add pattern detection
        from utils import detect_xss_patterns, analyze_payload_security
        patterns = detect_xss_patterns(payload)
        security_analysis = analyze_payload_security(payload)
        
        return {
            "original": payload,
            "cleaned": cleaned,
            "label": label,
            "confidence": round(prob, 4),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patterns": patterns,
            "security_analysis": security_analysis,
            "is_xss": label == "Malicious"
        }
    except Exception as e:
        logger.error(f"Error in detect_payload: {e}")
        return {
            "error": str(e),
            "original": payload,
            "cleaned": clean_payload(payload),
            "label": "Error",
            "confidence": 0.0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def parse_timestamp_date(timestamp):
    try:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").date().isoformat()
    except Exception:
        return timestamp.split(" ")[0] if " " in str(timestamp) else str(timestamp)

def save_log(entry):
    """Save detection log with duplicate prevention"""
    try:
        # Check for duplicate payload within time window
        if is_duplicate_payload(entry.get("original"), entry.get("timestamp")):
            logger.info(f"Duplicate payload detected, skipping log: {entry.get('original')[:50]}...")
            return False
        
        os.makedirs(os.path.dirname(LOG_PATH) or '.', exist_ok=True)
        logs = []
        if os.path.exists(LOG_PATH):
            try:
                with open(LOG_PATH, 'r') as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        
        logs.append({
            "timestamp": entry.get("timestamp"),
            "payload": entry.get("original"),
            "cleaned": entry.get("cleaned"),
            "attack_type": entry.get("label"),
            "confidence": entry.get("confidence")
        })
    
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(LOG_PATH, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Error saving log: {e}")
        return False

def build_summary_and_chart_data():
    """Build summary and chart data"""
    print("Building summary and chart data...")
    if not os.path.exists(LOG_PATH):
        print("Log file not found, returning empty data")
        return {}, {"dates": [], "malicious": [], "benign": []}
    
    try:
        with open(LOG_PATH, 'r') as f:
            logs = json.load(f)
        if not logs:
            print("No logs found, returning empty data")
            return {}, {"dates": [], "malicious": [], "benign": []}
        
        print(f"Found {len(logs)} log entries")

        # summary
        total_analyses = len(logs)
        total_attacks = len([x for x in logs if x['attack_type'] == 'Malicious'])
        all_attacks = [x['attack_type'] for x in logs]
        most_common = Counter(all_attacks).most_common(1)
        summary = {
            "total_analyses": total_analyses,
            "total_attacks": total_attacks,
            "last_result": logs[-1]['attack_type'] if logs else '-',
            "most_common_type": most_common[0][0] if most_common else '-'
        }

        # chart data per date
        counter = {}
        for entry in logs:
            date = parse_timestamp_date(entry['timestamp'])
            attack_type = entry['attack_type']
            key = (date, attack_type)
            counter[key] = counter.get(key, 0) + 1

        chart_dates = sorted({d for (d, _) in counter.keys()}, key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
        malicious_counts = [counter.get((d, "Malicious"), 0) for d in chart_dates]
        benign_counts = [counter.get((d, "Benign"), 0) for d in chart_dates]

        chart_data = {
            "dates": chart_dates,
            "malicious": malicious_counts,
            "benign": benign_counts
        }

        print(f"Chart data: {chart_data}")
        return summary, chart_data
    except Exception as e:
        logger.error(f"Error building summary: {e}")
        return {}, {"dates": [], "malicious": [], "benign": []}

def generate_pdf_report():
    """Generate PDF report from detection logs"""
    try:
        # Get data
        logs = get_detection_logs()
        if not logs:
            return None
        
        # Create chart
        line_chart_buffer = create_detection_charts(logs)
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                               leftMargin=1*inch, rightMargin=1*inch,
                               topMargin=1.2*inch, bottomMargin=1.5*inch)
        story = []
        
        # Formal Styles - Clean and Professional
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'FormalTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=20,
            alignment=1,  # Center
            textColor=colors.black,
            fontName='Helvetica-Bold',
            spaceBefore=10
        )
        subtitle_style = ParagraphStyle(
            'FormalSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            alignment=1,  # Center
            textColor=colors.black,
            fontName='Helvetica'
        )
        heading_style = ParagraphStyle(
            'FormalHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=20,
            textColor=colors.black,
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderPadding=5
        )
        normal_style = ParagraphStyle(
            'FormalNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=4,
            textColor=colors.black,
            fontName='Helvetica'
        )
        
        # Header - Simple and Clean
        story.append(Paragraph("SECURITY DETECTION REPORT", title_style))
        story.append(Spacer(1, 15))
        
        # Report Information - Simple Format
        info_data = [
            ['Report Date:', datetime.now().strftime('%B %d, %Y')],
            ['Report Time:', datetime.now().strftime('%I:%M %p')],
            ['System:', 'XSecure AI - Advanced Detection System'],
            ['Threshold:', f'{THRESHOLD}']
        ]
        
        info_table = Table(info_data, colWidths=[1.5*inch, 3.5*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black)
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 15))
        
        # Summary Statistics - Simple Format
        story.append(Paragraph("SUMMARY STATISTICS", heading_style))
        
        total_analyses = len(logs)
        total_attacks = len([x for x in logs if x['attack_type'] == 'Malicious'])
        total_benign = len([x for x in logs if x['attack_type'] == 'Benign'])
        
        # Calculate additional statistics
        if logs:
            dates = set()
            for log in logs:
                dates.add(log['timestamp'].split(' ')[0])
            total_analyses = len(dates)
        else:
            total_analyses = 0
        
        summary_data = [
            ['Total Analyses', f'{total_analyses:,}'],
            ['Malicious Detections', f'{total_attacks:,}'],
            ['Benign Detections', f'{total_benign:,}'],
            ['Detection Rate', f"{(total_attacks/total_analyses*100):.1f}%" if total_analyses > 0 else "0%"]
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 15))
        
        # Chart - Single Line Chart Only
        if line_chart_buffer:
            story.append(Paragraph("DETECTION TRENDS", heading_style))
            
            # Single Line Chart
            line_img = Image(line_chart_buffer, width=5.5*inch, height=3*inch)
            line_img.hAlign = 'CENTER'
            story.append(line_img)
            story.append(Spacer(1, 15))
        
        # Daily Breakdown - Simple Format
        if logs:
            story.append(Paragraph("DAILY BREAKDOWN", heading_style))
            
            # Group by date
            daily_stats = {}
            for log in logs:
                date = log['timestamp'].split(' ')[0]
                if date not in daily_stats:
                    daily_stats[date] = {'malicious': 0, 'benign': 0}
                daily_stats[date][log['attack_type'].lower()] += 1
            
            # Create daily table
            daily_data = [['Date', 'Malicious', 'Benign', 'Total']]
            for date in sorted(daily_stats.keys()):
                stats = daily_stats[date]
                total = stats['malicious'] + stats['benign']
                daily_data.append([date, f'{stats["malicious"]:,}', f'{stats["benign"]:,}', f'{total:,}'])
            
            daily_table = Table(daily_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            daily_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.black)
            ]))
            
            story.append(daily_table)
            story.append(Spacer(1, 15))
        
        # Recent Detections - Simple Format
        story.append(Paragraph("RECENT DETECTIONS", heading_style))
        
        if logs:
            recent_logs = logs[-10:]  # Last 10 entries
            table_data = [['Date', 'Payload', 'Type', 'Confidence', 'Timestamp']]
            for log in recent_logs:
                date = log['timestamp'].split(' ')[0]
                payload = log['payload'][:25] + "..." if len(log['payload']) > 25 else log['payload']
                attack_type = log['attack_type']
                confidence = f"{log['confidence']:.3f}"
                timestamp = log['timestamp']
                table_data.append([date, payload, attack_type, confidence, timestamp])
            recent_table = Table(table_data, colWidths=[1*inch, 2.8*inch, 1*inch, 1*inch, 1.5*inch])
            recent_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(recent_table)
        else:
            story.append(Paragraph("No detection data available.", normal_style))
        
        # Security Recommendations - Simple Format
        story.append(Spacer(1, 15))
        story.append(Paragraph("SECURITY RECOMMENDATIONS", heading_style))
        
        recommendations = [
            "Implement comprehensive input validation and sanitization",
            "Deploy Content Security Policy (CSP) headers",
            "Maintain updated frameworks and libraries",
            "Establish continuous monitoring and logging",
            "Perform regular security assessments and audits"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", normal_style))
        
        # Add footer as content
        story.append(Spacer(1, 20))
        footer_style = ParagraphStyle(
            'Footer',
            parent=normal_style,
            fontSize=7,
            textColor=colors.HexColor('#95a5a6'),
            alignment=0  # Left align
        )
        story.append(Paragraph("Generated by XSecure AI - Advanced Security Detection System", footer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return None

def create_detection_charts(logs):
    """Create single line chart for the PDF report"""
    try:
        if not logs:
            return None
        
        # Prepare data
        daily_stats = {}
        for log in logs:
            date = log['timestamp'].split(' ')[0]
            if date not in daily_stats:
                daily_stats[date] = {'malicious': 0, 'benign': 0}
            daily_stats[date][log['attack_type'].lower()] += 1
        
        dates = sorted(daily_stats.keys())
        malicious_counts = [daily_stats[date]['malicious'] for date in dates]
        benign_counts = [daily_stats[date]['benign'] for date in dates]
        
        # Create line chart
        plt.figure(figsize=(10, 6))
        plt.plot(dates, malicious_counts, 'r-', linewidth=2, marker='o', markersize=6, label='Malicious')
        plt.plot(dates, benign_counts, 'g-', linewidth=2, marker='s', markersize=6, label='Benign')
        plt.title('Security Detection Trends Over Time', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Total Detections', fontsize=12, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save line chart
        line_chart_buffer = BytesIO()
        plt.savefig(line_chart_buffer, format='png', dpi=300, bbox_inches='tight')
        line_chart_buffer.seek(0)
        plt.close()
        
        return line_chart_buffer
        
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return None


# ========== Routes ==========
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        payload = request.form.get('payload', '').strip()
        if not payload:
            return render_template('index.html', error="Please enter a payload to analyze.")
        
        # Validate payload
        is_valid, error_msg = validate_payload(payload)
        if not is_valid:
            return render_template('index.html', error=error_msg)
        
        result = detect_payload(payload)
        if 'error' not in result:
            # Only save log if not duplicate and broadcast
            if save_log(result):
                socketio.emit('new_detection', result, include_self=False)
        
        return render_template('index.html', original=result['original'],
                               cleaned=result['cleaned'],
                               label=result['label'],
                               confidence=f"{result['confidence']}")
    return render_template('index.html')

@app.route('/history')
def history():
    summary, chart_data = build_summary_and_chart_data()
    return render_template('dashboard.html', chart_data=chart_data, summary=summary)

# ----------------- ROUTE REPORT -----------------
def get_detection_logs():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

@app.route('/report')
def report():
    logs = get_detection_logs()
    from collections import defaultdict
    pivoted = defaultdict(lambda: {"Malicious": 0, "Benign": 0})
    for log in logs:
        date = parse_timestamp_date(log['timestamp'])
        pivoted[date][log['attack_type']] += 1

    table_data = [
        {"date": date, "malicious": vals["Malicious"], "benign": vals["Benign"]}
        for date, vals in sorted(pivoted.items(), reverse=True)
    ]

    summary = {
        "total_analyses": len(logs),
        "total_attacks": len([log for log in logs if log['attack_type'] == 'Malicious']),
        "most_common_type": max(set([log['attack_type'] for log in logs]),
                                key=[log['attack_type'] for log in logs].count) if logs else '-'
    }

    latest = logs[-1] if logs else {}
    return render_template(
        'report.html',
        table_data=table_data,
        summary=summary,
        original=latest.get('payload', '-'),
        cleaned=latest.get('cleaned', '-'),
        label=latest.get('attack_type', '-'),
        confidence=latest.get('confidence', '-'),
        timestamp=latest.get('timestamp', '-')
    )

@app.route('/download_report')
def download_report():
    """Download PDF report"""
    try:
        pdf_buffer = generate_pdf_report()
        if pdf_buffer is None:
            return jsonify({"error": "Failed to generate PDF report"}), 500
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"xss_detection_report_{timestamp}.pdf"
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        return jsonify({"error": "Failed to download report"}), 500

@app.route('/download_logs/<date>/<fmt>')
def download_logs(date, fmt):
    logs = get_detection_logs()
    filtered = [log for log in logs if parse_timestamp_date(log['timestamp']) == date]
    if not filtered:
        return jsonify({"error": "No logs found for this date"}), 404
    
    # ==== CSV ====
    if fmt == "csv":
        import csv
        from io import StringIO
        si = StringIO()
        cw = csv.writer(si)
        cw.writerow(["timestamp", "payload", "cleaned", "attack_type", "confidence"])
        for log in filtered:
            cw.writerow([
                log.get("timestamp"),
                log.get("payload"),
                log.get("cleaned"),
                log.get("attack_type"),
                log.get("confidence")
            ])
        output = si.getvalue()
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename=logs_{date}.csv"}
        )

    # ==== JSON ====
    elif fmt == "json":
        return Response(
            json.dumps(filtered, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename=logs_{date}.json"}
        )

    # ==== PDF ====
    elif fmt == "pdf":
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from io import BytesIO

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        story.append(Paragraph(f"Detection Logs for {date}", styles['Title']))
        story.append(Spacer(1, 12))

        data = [["Timestamp", "Payload", "Type", "Confidence"]]
        for log in filtered:
            data.append([
                log["timestamp"],
                (log["payload"][:30] + "...") if len(log["payload"]) > 30 else log["payload"],
                log["attack_type"],
                str(log["confidence"])
            ])

        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black)
        ]))

        story.append(table)
        doc.build(story)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True,
                         download_name=f"logs_{date}.pdf",
                         mimetype="application/pdf")
    else:
        return jsonify({"error": "Invalid format"}), 400

# ========== API Endpoints ==========
@app.route('/api/detect', methods=['POST'])
def detect_api():
    """Main API endpoint for XSS detection"""
    try:
        data = request.get_json() or {}
        payload = data.get('payload', '')
        
        if not payload:
            return jsonify({"error": "Payload is required"}), 400
        
        # Validate payload
        is_valid, error_msg = validate_payload(payload)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        result = detect_payload(payload)
        if 'error' not in result:
            # Only save log if not duplicate and broadcast
            if save_log(result):
                socketio.emit('new_detection', result, include_self=False)
        
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/batch-detect', methods=['POST'])
def batch_detect_api():
    """Batch detection API for multiple payloads"""
    try:
        data = request.get_json() or {}
        payloads = data.get('payloads', [])
        
        if not payloads or not isinstance(payloads, list):
            return jsonify({"error": "Payloads must be a non-empty array"}), 400
        
        if len(payloads) > 100:  # Limit batch size
            return jsonify({"error": "Maximum 100 payloads per batch"}), 400
        
        results = []
        for payload in payloads:
            if not payload:
                results.append({"error": "Empty payload"})
                continue
            
            is_valid, error_msg = validate_payload(payload)
            if not is_valid:
                results.append({"error": error_msg})
                continue
            
            result = detect_payload(payload)
            if 'error' not in result:
                # Only save log if not duplicate
                save_log(result)
            results.append(result)
        
        socketio.emit('batch_detection_complete', {"count": len(results)}, include_self=False)
        return jsonify({"results": results}), 200
    except Exception as e:
        logger.error(f"Batch API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/status', methods=['GET'])
def status_api():
    """Health check and status API"""
    try:
        model_status = "loaded" if model is not None else "not_loaded"
        tokenizer_status = "loaded" if tokenizer is not None else "not_loaded"
        
        summary, _ = build_summary_and_chart_data()
        
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "tokenizer_status": tokenizer_status,
            "threshold": THRESHOLD,
            "max_length": MAX_LEN,
            "total_analyses": summary.get("total_analyses", 0),
            "total_attacks": summary.get("total_attacks", 0)
        }), 200
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def logs_api():
    """Get recent detection logs"""
    try:
        limit = request.args.get('limit', 50, type=int)
        limit = min(limit, 1000)  # Max 1000 logs
        
        logs = get_detection_logs()
        recent_logs = logs[-limit:] if logs else []
        
        return jsonify({
            "logs": recent_logs,
            "total": len(logs),
            "returned": len(recent_logs)
        }), 200
    except Exception as e:
        logger.error(f"Logs API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/clear-logs', methods=['POST'])
def clear_logs_api():
    """Clear all detection logs"""
    try:
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
        return jsonify({"message": "Logs cleared successfully"}), 200
    except Exception as e:
        logger.error(f"Clear logs error: {e}")
        return jsonify({"error": "Failed to clear logs"}), 500

@app.route('/api/latest-log', methods=['GET'])
def latest_log_api():
    """Get the most recent detection log"""
    try:
        logs = get_detection_logs()
        if not logs:
            return jsonify({"log": None}), 200
        
        latest = logs[-1]
        return jsonify({"log": latest}), 200
    except Exception as e:
        logger.error(f"Latest log API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ========== Socket.IO Handlers ==========
@socketio.on('submit_payload')
def handle_submit_payload(data):
    """Handle real-time payload submission dengan validasi session code"""
    sid = request.sid
    # Cek apakah client sudah join dengan session code valid
    if sid not in client_sessions:
        emit('detection_result', {"error": "Unauthorized: Session code required. Please join first."})
        return
    try:
        payload = data.get('payload', '')
        if not payload:
            emit('detection_result', {"error": "Please enter a payload to analyze"})
            return
        # Validate payload
        is_valid, error_msg = validate_payload(payload)
        if not is_valid:
            emit('detection_result', {"error": error_msg})
            return
        result = detect_payload(payload)
        if 'error' not in result:
            if save_log(result):
                socketio.emit('new_detection', result, include_self=False)
        emit('detection_result', result)
    except Exception as e:
        logger.error(f"Socket error: {e}")
        emit('detection_result', {"error": "Internal server error occurred"})

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {"message": "Connected to XSS Detection System"})

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in client_sessions:
        logger.info(f"Client disconnected: {sid} (session {client_sessions[sid]['session_code']})")
        del client_sessions[sid]
    else:
        logger.info(f"Client disconnected: {sid}")

# ========== Error Handlers ==========
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ========== Run ==========
if __name__ == '__main__':
    print("🚀 Starting XSS Detection Server...")
    print(f"📊 Model Path: {MODEL_PATH}")
    print(f"🔧 Threshold: {THRESHOLD}")
    print(f"🌐 Server will run on: http://0.0.0.0:5000")
    print(f"📡 WebSocket enabled for real-time detection")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
