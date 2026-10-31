import os
import cv2
import json
import base64
import webbrowser
import threading
import numpy as np
from datetime import datetime, date
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file, session
from werkzeug.utils import secure_filename

from config import (
    SECRET_KEY, UPLOAD_FOLDER, CANDIDATE_PHOTOS_DIR, AADHAAR_PHOTOS_DIR,
    TEST_VIDEOS_DIR, PROCESSED_VIDEOS_DIR, REPORTS_DIR,
    VIDEO_HELMET_PASS_RATIO, VIDEO_VIOLATION_RATIO
)
from database.db import (
    init_db, register_candidate, get_candidate, 
    get_all_candidates, save_evaluation, get_evaluation,
    create_user, authenticate_user, get_user, generate_candidate_id
)
from models.facenet_auth import facenet_authenticator
from models.yolo_detector import yolo_detector
from models.lane_detector import lane_detector
from models.pose_analyzer import pose_analyzer
from models.ml_evaluator import ml_evaluator
from utils.pdf_generator import generate_pdf_report

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize DB tables on startup
init_db()

@app.route('/')
def index():
    if not session.get('user_id'):
        return redirect(url_for('login'))
        
    candidates = get_all_candidates()
    # Calculate stats
    total_cands = len(candidates)
    
    # Calculate test statistics
    from database.db import get_db_connection
    c = get_db_connection()
    evals = c.execute('SELECT * FROM evaluations').fetchall()
    c.close()
    
    pass_count = sum(1 for e in evals if e['overall_result'].upper() == 'PASS')
    fail_count = sum(1 for e in evals if e['overall_result'].upper() == 'FAIL')

    stats = {
        'total_candidates': total_cands,
        'total_evaluations': len(evals),
        'pass_count': pass_count,
        'fail_count': fail_count
    }

    return render_template('index.html', candidates=candidates, stats=stats)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        
        user = authenticate_user(email, password)
        if user:
            session['user_id'] = user['id']
            session['user_name'] = user['full_name']
            session['user_email'] = user['email']
            session['user_mobile'] = user['mobile']
            flash(f'Welcome, {user["full_name"]}! Please complete Step 1: Candidate Details.', 'success')
            return redirect(url_for('register'))
        else:
            flash('Invalid Email/User ID or Password. Please try again.', 'danger')
            return redirect(url_for('login'))
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        mobile = request.form.get('mobile', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        if password != confirm_password:
            flash('Passwords do not match. Please try again.', 'danger')
            return redirect(url_for('signup'))
            
        success, msg_or_id = create_user(full_name, email, mobile, password)
        if success:
            flash('Registration successful. Please log in to continue.', 'success')
            return redirect(url_for('login'))
        else:
            flash(f'Registration failed: {msg_or_id}', 'danger')
            return redirect(url_for('signup'))
            
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

def get_max_dob_for_18(today=None):
    """
    Returns the maximum Date of Birth (date object) such that the candidate
    is at least 18 years old today. Correctly handles leap-year birthdays.
    """
    if today is None:
        today = date.today()
    try:
        return today.replace(year=today.year - 18)
    except ValueError:
        # Occurs if today is February 29 (leap year) and (today.year - 18) is not a leap year.
        # On Feb 29, someone born on Feb 28, 18 years ago is 18 years and 1 day old.
        return today.replace(year=today.year - 18, day=28)

def calculate_age(dob_date, today=None):
    """
    Calculates exact age in completed years, correctly handling leap years.
    """
    if today is None:
        today = date.today()
    return today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if not session.get('user_id'):
        flash('Please log in first to begin Step 1: Candidate Details.', 'warning')
        return redirect(url_for('login'))

    max_dob_str = get_max_dob_for_18().strftime('%Y-%m-%d')

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        aadhaar = request.form.get('aadhaar', '').strip()
        dob = request.form.get('dob', '').strip()
        gender = request.form.get('gender', '').strip()
        address = request.form.get('address', '').strip()
        mobile = request.form.get('mobile', '').strip()
        email = request.form.get('email', '').strip()

        # Validate mandatory fields
        if not name:
            flash('Candidate Name is required.', 'danger')
            return redirect(url_for('register'))

        if not dob:
            flash('Date of birth is required.', 'danger')
            return redirect(url_for('register'))

        # Strict Age Validation (Must be at least 18 years old based on exact DOB)
        try:
            dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
            today = date.today()
            age = calculate_age(dob_date, today)
        except (ValueError, TypeError):
            flash('Invalid Date of Birth format.', 'danger')
            return redirect(url_for('register'))

        if age < 18:
            flash('You must be at least 18 years old to register.', 'danger')
            return redirect(url_for('register'))

        if not gender:
            flash('Gender is required.', 'danger')
            return redirect(url_for('register'))

        if not mobile:
            flash('Mobile Number is required.', 'danger')
            return redirect(url_for('register'))

        if not email:
            flash('Email ID is required.', 'danger')
            return redirect(url_for('register'))

        if not address:
            flash('Residential Address is required.', 'danger')
            return redirect(url_for('register'))

        # Aadhaar Number Validation (Exactly 12 digits)
        if not aadhaar or len(aadhaar) != 12 or not aadhaar.isdigit():
            flash('Aadhaar Number must contain exactly 12 digits.', 'danger')
            return redirect(url_for('register'))

        # Aadhaar Card Image File Upload & Validation
        aadhaar_file = request.files.get('aadhaar_image')
        if not aadhaar_file or aadhaar_file.filename == '':
            flash('Please upload your Aadhaar Card image.', 'danger')
            return redirect(url_for('register'))

        ext = os.path.splitext(aadhaar_file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            flash('Invalid Aadhaar Card image format. Only JPG, JPEG, and PNG files are accepted.', 'danger')
            return redirect(url_for('register'))

        aadhaar_file.seek(0, os.SEEK_END)
        aadhaar_file_size = aadhaar_file.tell()
        aadhaar_file.seek(0)
        if aadhaar_file_size > 5 * 1024 * 1024:
            flash('Aadhaar Card image file size must not exceed 5 MB.', 'danger')
            return redirect(url_for('register'))

        candidate_id = generate_candidate_id()

        # Save Aadhaar Image File
        aadhaar_filename = f"aadhaar_{candidate_id}_" + secure_filename(aadhaar_file.filename)
        aadhaar_filepath = os.path.join(AADHAAR_PHOTOS_DIR, aadhaar_filename)
        aadhaar_file.save(aadhaar_filepath)
        aadhaar_photo_path = f"static/uploads/aadhaar/{aadhaar_filename}"

        photo_path = ""
        captured_data = request.form.get('captured_photo_data')

        # Handle Photo Upload or Base64 WebCam capture
        if captured_data and captured_data.startswith('data:image'):
            header, encoded = captured_data.split(",", 1)
            data = base64.b64decode(encoded)
            filename = f"photo_{candidate_id}.jpg"
            filepath = os.path.join(CANDIDATE_PHOTOS_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(data)
            photo_path = f"static/uploads/photos/{filename}"
        elif 'photo' in request.files and request.files['photo'].filename != '':
            file = request.files['photo']
            filename = f"photo_{candidate_id}_" + secure_filename(file.filename)
            filepath = os.path.join(CANDIDATE_PHOTOS_DIR, filename)
            file.save(filepath)
            photo_path = f"static/uploads/photos/{filename}"

        if not photo_path:
            flash('Please upload or capture a passport-size photo.', 'danger')
            return redirect(url_for('register'))

        # Extract FaceNet Embedding
        abs_photo_path = os.path.join(app.root_path, photo_path)
        face_embedding = facenet_authenticator.extract_embedding(abs_photo_path)

        candidate_data = {
            'candidate_id': candidate_id,
            'name': name,
            'aadhaar': aadhaar,
            'aadhaar_photo_path': aadhaar_photo_path,
            'dob': dob,
            'gender': gender,
            'address': address,
            'mobile': mobile,
            'email': email,
            'photo_path': photo_path
        }

        success, err = register_candidate(candidate_data, face_embedding)
        if success:
            flash(f'Registration Successful! Your Candidate ID is {candidate_id}. Proceeding to Step 2: Face Verification.', 'success')
            return redirect(url_for('authenticate', candidate_id=candidate_id))
        else:
            if 'UNIQUE constraint failed: candidates.aadhaar' in str(err):
                flash('A candidate with this Aadhaar Number has already been registered.', 'danger')
            else:
                flash(f'Registration failed: {err}', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html', max_dob=max_dob_str)

@app.route('/authenticate')
def authenticate():
    if not session.get('user_id'):
        flash('Please log in first to access Step 2: Face Verification.', 'warning')
        return redirect(url_for('login'))

    candidate_id = request.args.get('candidate_id', '')
    candidates = get_all_candidates()
    return render_template('authenticate.html', candidates=candidates, selected_candidate_id=candidate_id)

@app.route('/api/authenticate_face', methods=['POST'])
def api_authenticate_face():
    try:
        data = request.json
        if not data:
            return jsonify({'auth_success': False, 'distance': 1.0, 'confidence': 0.0, 'message': 'Invalid request payload'}), 400

        candidate_id = data.get('candidate_id')
        image_data_url = data.get('image_data')

        if not candidate_id:
            return jsonify({'auth_success': False, 'distance': 1.0, 'confidence': 0.0, 'message': 'No candidate selected'}), 400

        candidate = get_candidate(candidate_id)
        if not candidate:
            return jsonify({'auth_success': False, 'distance': 1.0, 'confidence': 0.0, 'message': 'Candidate not found'}), 404

        if not image_data_url or ',' not in image_data_url:
            return jsonify({'auth_success': False, 'distance': 1.0, 'confidence': 0.0, 'message': 'No face photo captured or uploaded'}), 400

        # Decode base64 webcam frame or uploaded photo
        try:
            header, encoded = image_data_url.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as err:
            return jsonify({'auth_success': False, 'distance': 1.0, 'confidence': 0.0, 'message': f'Failed to decode image data: {str(err)}'}), 400

        if frame is None or frame.size == 0:
            return jsonify({'auth_success': False, 'distance': 1.0, 'confidence': 0.0, 'message': 'Captured image is empty or invalid'}), 400

        reg_embedding = candidate.get('face_embedding')
        if not reg_embedding:
            # Generate on the fly from photo if embedding missing
            photo_abs_path = os.path.join(app.root_path, candidate['photo_path'])
            if not os.path.exists(photo_abs_path):
                return jsonify({'auth_success': False, 'distance': 1.0, 'confidence': 0.0, 'message': 'Registered candidate photo file missing on server'}), 404
            reg_embedding = facenet_authenticator.extract_embedding(photo_abs_path)

        auth_passed, distance, confidence = facenet_authenticator.compare_faces(reg_embedding, frame)

        if not auth_passed:
            return jsonify({
                'auth_success': False,
                'distance': round(distance, 4),
                'confidence': confidence,
                'message': f'Facial match score ({confidence}%) is below the required 70% threshold. Candidate verification rejected.'
            })

        return jsonify({
            'auth_success': True,
            'distance': round(distance, 4),
            'confidence': confidence,
            'message': f'Candidate authenticated successfully with {confidence}% match confidence.'
        })
    except Exception as e:
        print(f"Error in api_authenticate_face: {e}")
        return jsonify({
            'auth_success': False,
            'distance': 1.0,
            'confidence': 0.0,
            'message': f"Server error during facial verification: {str(e)}"
        }), 500

@app.route('/analysis')
def analysis():
    if not session.get('user_id'):
        flash('Please log in first to access Step 3: Driving Test Video Verification.', 'warning')
        return redirect(url_for('login'))

    candidate_id = request.args.get('candidate_id', '')
    candidates = get_all_candidates()
    return render_template('analysis.html', candidates=candidates, selected_candidate_id=candidate_id)

@app.route('/api/analyze_video', methods=['POST'])
def api_analyze_video():
    try:
        candidate_id = request.form.get('candidate_id')
        if not candidate_id:
            return jsonify({'error': 'No candidate selected'}), 400

        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400

        filename = f"test_{candidate_id}_" + secure_filename(video_file.filename)
        video_save_path = os.path.join(TEST_VIDEOS_DIR, filename)
        video_file.save(video_save_path)

        candidate = get_candidate(candidate_id)
        if not candidate:
            return jsonify({'error': 'Candidate not found'}), 404

        # Process Video Frame-by-Frame
        cap = cv2.VideoCapture(video_save_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        lane_detector.reset()

        frame_count = 0
        total_rider_frames = 0
        reliable_rider_frames = 0
        helmet_pass_count = 0
        no_helmet_count = 0
        other_headwear_count = 0
        helmet_review_count = 0
        helmet_confidences = []

        lane_violation_count = 0
        foot_touch_count = 0
        unstable_balance_count = 0
        max_lean_angle = 0.0

        sample_frame_url = ""

        # Process sample frames (stride = 3 for efficiency)
        stride = 3
        last_annotated_frame = None
        best_annotated_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % stride != 0:
                continue

            # 1. YOLOv8 Detections (Rider, Motorcycle, Helmet, Ground line)
            yolo_dets, yolo_frame = yolo_detector.detect_objects(frame)
            if yolo_dets['rider_found']:
                total_rider_frames += 1
                h_cls = yolo_dets.get('helmet_class', 'none')
                h_stat = yolo_dets.get('helmet_status', 'FAIL')
                h_conf = yolo_dets.get('helmet_conf', 0.0)

                if h_cls == 'helmet' and h_stat == 'PASS':
                    helmet_pass_count += 1
                    reliable_rider_frames += 1
                    helmet_confidences.append(h_conf)
                elif h_cls == 'other_headwear':
                    other_headwear_count += 1
                    reliable_rider_frames += 1
                elif h_cls == 'no_helmet':
                    no_helmet_count += 1
                    reliable_rider_frames += 1
                elif h_stat == 'Needs Review':
                    helmet_review_count += 1
                    if h_cls == 'helmet' and h_conf > 0:
                        helmet_confidences.append(h_conf)

            # 2. OpenCV Lane Boundary Detection & Motorcycle Tracking
            # Uses YOLO motorcycle detection and tracks motorcycle relative to boundaries
            lane_viol, lane_frame = lane_detector.analyze_lane(
                yolo_frame,
                rider_bbox=yolo_dets['rider_bbox'],
                motorcycle_bbox=yolo_dets['motorcycle_bbox'],
                frame_idx=frame_count,
                fps=fps
            )
            if lane_viol:
                lane_violation_count += 1

            # 3. MediaPipe Pose Analysis (Foot touch & balance angle)
            pose_info, final_frame = pose_analyzer.analyze_pose(lane_frame, yolo_dets['ground_line_y'])
            if pose_info['foot_touch']:
                foot_touch_count += 1
            if pose_info['balance_status'] == 'Unstable':
                unstable_balance_count += 1
            if abs(pose_info['lean_angle']) > max_lean_angle:
                max_lean_angle = abs(pose_info['lean_angle'])

            if yolo_dets['rider_found']:
                best_annotated_frame = final_frame
            last_annotated_frame = final_frame

        cap.release()

        total_sampled_frames = max(1, frame_count // stride)

        # Save a sample annotated frame for frontend display
        frame_to_save = best_annotated_frame if best_annotated_frame is not None else last_annotated_frame
        if frame_to_save is not None:
            sample_img_name = f"sample_{candidate_id}.jpg"
            sample_img_path = os.path.join(PROCESSED_VIDEOS_DIR, sample_img_name)
            cv2.imwrite(sample_img_path, frame_to_save)
            sample_frame_url = f"/static/uploads/processed/{sample_img_name}"

        # Evaluate Skill Parameters — Practical & Tolerant Evaluation (Not Overly Strict)
        auth_status = 1

        # 1. Helmet Compliance (Compulsory Requirement)
        # Evaluated across multiple reliable frames:
        # - Confidently detected genuine motorcycle helmet (conf >= threshold): PASS
        # - Helmet detected with conf < 85%: Needs Review
        # - Confidently detected uncovered head: FAIL
        # - Confidently detected cap or other non-protective headwear: FAIL
        # - Blurred, hidden, or uncertain head throughout: NEEDS REVIEW
        helmet_notes = []
        actual_helmet_conf = 0.0

        if total_rider_frames == 0:
            helmet_state = 'Needs Review'
            helmet_status = 0
            helmet_notes.append("No rider detected in video frames for helmet verification.")
        elif reliable_rider_frames < 3:
            helmet_state = 'Needs Review'
            helmet_status = 0
            helmet_notes.append("Head region was blurred, distant, or uncertain throughout video frames (Manual Safety Review Required).")
        else:
            helmet_pass_ratio = helmet_pass_count / reliable_rider_frames
            violation_ratio = (other_headwear_count + no_helmet_count) / reliable_rider_frames
            no_helmet_ratio = no_helmet_count / reliable_rider_frames
            other_headwear_ratio = other_headwear_count / reliable_rider_frames

            # Case A: Confirmed uncovered head (bare face/scalp without helmet)
            if no_helmet_count >= 3 and no_helmet_ratio >= 0.35 and no_helmet_count > helmet_pass_count:
                helmet_state = 'FAIL'
                helmet_status = 0
                actual_helmet_conf = 0.0
                helmet_notes.append("Uncovered head detected - Rider was not wearing a protective motorcycle helmet.")
            # Case B: Confirmed non-protective headwear (Cap / Hat / Cloth)
            elif other_headwear_count >= 3 and other_headwear_ratio >= 0.35 and other_headwear_count > helmet_pass_count:
                helmet_state = 'FAIL'
                helmet_status = 0
                actual_helmet_conf = 0.0
                helmet_notes.append("Non-protective headwear (Cap/Hat/Cloth) confirmed on rider (Motorcycle Safety Helmet Required).")
            # Case C: Confidently detected genuine motorcycle helmet across reliable frames
            elif helmet_pass_count >= 3 and helmet_pass_ratio >= 0.50 and violation_ratio < 0.35:
                avg_conf = round(float(np.mean(helmet_confidences)), 2) if helmet_confidences else 0.0
                actual_helmet_conf = avg_conf
                if actual_helmet_conf >= yolo_detector.helmet_conf_threshold:
                    helmet_state = 'PASS'
                    helmet_status = 1
                else:
                    helmet_state = 'Needs Review'
                    helmet_status = 0
                    helmet_notes.append(f"Helmet detection confidence ({int(actual_helmet_conf * 100)}%) is below the {int(yolo_detector.helmet_conf_threshold * 100)}% threshold (Manual Safety Review Required).")
            # Case D: Inconclusive, ambiguous, or mixed detections
            else:
                helmet_state = 'Needs Review'
                helmet_status = 0
                actual_helmet_conf = round(float(np.mean(helmet_confidences)), 2) if helmet_confidences else 0.0
                helmet_notes.append("Inconclusive helmet detection across video frames (Manual Safety Review Required).")

        # 2. Lane Discipline (Straight Driving Lane Tracking)
        # Evaluates confirmed boundary contacts across video frames
        lane_summary = lane_detector.get_summary()
        lane_state = lane_summary['lane_state']
        lane_status = lane_summary['lane_status']
        lane_violations = lane_summary['violations']
        normalized_lane_count = len(lane_violations)

        lane_notes = []
        if lane_state == 'FAIL':
            for v in lane_violations:
                lane_notes.append(f"LANE_BOUNDARY_VIOLATION: Motorcycle touched/crossed {v['boundary']} boundary at {v['timestamp']}")
        elif lane_state in ['NEEDS_REVIEW', 'Needs Review']:
            lane_notes.append(f"Lane Tracking Inconclusive: {lane_summary['remarks']} (Manual Review Required)")

        # 3. Body Balance (PASS if rider generally stable; fail only on major instability)
        unstable_ratio = unstable_balance_count / max(1, total_sampled_frames)
        if unstable_ratio < 0.35:
            balance_state = 'PASS'
            balance_status = 1
        else:
            balance_state = 'FAIL'
            balance_status = 0

        # 4. Foot Touch Detection (Only confirmed ground contacts count)
        actual_foot_count = foot_touch_count // 3 if foot_touch_count >= 3 else (1 if foot_touch_count >= 2 else 0)
        if actual_foot_count == 0:
            foot_touch_state = 'PASS'
            foot_touch_status = 1
        else:
            foot_touch_state = 'FAIL'
            foot_touch_status = 0

        # 5. Braking Control (Tolerant stopping check: PASS for controlled stopping & normal deceleration)
        if unstable_ratio < 0.50 and max_lean_angle <= 65.0:
            braking_state = 'PASS'
            braking_status = 1
        else:
            braking_state = 'FAIL'
            braking_status = 0

        # 4. Machine Learning Driving Performance Prediction (Random Forest)
        ml_res = ml_evaluator.evaluate(
            auth_status=auth_status,
            helmet_status=helmet_status,
            lane_status=lane_status,
            foot_touch_status=foot_touch_status,
            balance_status=balance_status,
            lane_violation_count=normalized_lane_count,
            foot_touch_count=actual_foot_count,
            max_lean_angle=max_lean_angle,
            braking_status=braking_status,
            helmet_state=helmet_state,
            balance_state=balance_state,
            foot_touch_state=foot_touch_state,
            lane_state=lane_state,
            braking_state=braking_state
        )

        # Merge specific helmet violation/review notes into mistakes list
        for note in reversed(helmet_notes):
            if note not in ml_res['mistakes']:
                ml_res['mistakes'].insert(0, note)

        # Merge specific lane boundary violation notes into mistakes list
        for note in lane_notes:
            if note not in ml_res['mistakes']:
                ml_res['mistakes'].append(note)

        eval_record = {
            'candidate_id': candidate_id,
            'auth_status': auth_status,
            'helmet_status': helmet_status,
            'lane_status': lane_status,
            'foot_touch_status': foot_touch_status,
            'balance_status': balance_status,
            'overall_result': ml_res['final_result'],
            'rf_prediction': ml_res['rf_prediction'],
            'overall_score': ml_res['overall_score'],
            'mistakes': ml_res['mistakes'],
            'video_path': f"static/uploads/videos/{filename}",
            'processed_video_path': sample_frame_url,
            'report_path': ''
        }

        # Save evaluation record to DB
        eval_id = save_evaluation(eval_record)
        eval_record['id'] = eval_id

        # 5. Generate Automated PDF Report
        pdf_abs_path = generate_pdf_report(eval_record, candidate)

        return jsonify({
            'evaluation_id': eval_id,
            'overall_result': ml_res['final_result'],
            'overall_score': ml_res['overall_score'],
            'rf_prediction': ml_res['rf_prediction'],
            'helmet_status': bool(helmet_status),
            'helmet_confidence': actual_helmet_conf,
            'lane_status': bool(lane_status),
            'foot_touch_status': bool(foot_touch_status),
            'balance_status': bool(balance_status),
            'braking_status': bool(braking_status),
            'helmet_state': helmet_state,
            'balance_state': balance_state,
            'foot_touch_state': foot_touch_state,
            'lane_state': lane_state,
            'braking_state': braking_state,
            'lane_violation_count': normalized_lane_count,
            'lane_violations': lane_violations,
            'foot_touch_count': actual_foot_count,
            'unstable_balance_count': unstable_balance_count,
            'mistakes': ml_res['mistakes'],
            'sample_frame_url': sample_frame_url
        })
    except Exception as e:
        print(f"Error in api_analyze_video: {e}")
        return jsonify({'error': f"Video processing error: {str(e)}"}), 500

@app.route('/report/<int:eval_id>')
def report(eval_id):
    if not session.get('user_id'):
        flash('Please log in first to view reports.', 'warning')
        return redirect(url_for('login'))

    evaluation = get_evaluation(eval_id)
    if not evaluation:
        flash('Evaluation record not found.', 'danger')
        return redirect(url_for('index'))
    return render_template('report.html', evaluation=evaluation)

@app.route('/download_report/<int:eval_id>')
def download_report(eval_id):
    if not session.get('user_id'):
        flash('Please log in first to download reports.', 'warning')
        return redirect(url_for('login'))

    evaluation = get_evaluation(eval_id)
    if not evaluation:
        flash('Evaluation record not found.', 'danger')
        return redirect(url_for('index'))

    candidate = get_candidate(evaluation['candidate_id'])
    pdf_path = generate_pdf_report(evaluation, candidate)

    return send_file(pdf_path, as_attachment=True)

def open_browser():
    try:
        webbrowser.open("http://localhost:5000/login")
    except Exception as e:
        print(f"Could not automatically open browser: {e}")

if __name__ == '__main__':
    print("=" * 75)
    print("  🚀 AI Driving Skill Validation System Running!")
    print("  👉 Login Page (Start Here):       http://localhost:5000/login")
    print("  👉 Step 1: Candidate Register:    http://localhost:5000/register")
    print("  👉 Step 2: Face Verification:     http://localhost:5000/authenticate")
    print("  👉 Step 3: Video Test Analysis:   http://localhost:5000/analysis")
    print("=" * 75)

    # Launch browser automatically on startup
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        threading.Timer(1.2, open_browser).start()
    elif not os.environ.get('WERKZEUG_RUN_MAIN'):
        threading.Timer(1.2, open_browser).start()

    app.run(host='0.0.0.0', port=5000, debug=True)
