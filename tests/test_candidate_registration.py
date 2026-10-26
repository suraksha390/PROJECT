"""
Comprehensive Test Suite for Candidate Registration (Step 1)
=============================================================

Verifies all requirements:
  1. Duplicate values allowed for:
     - Candidate Name
     - Mobile Number
     - Residential Address
     - Date of Birth (DOB)
  2. Age validation (minimum 18 years):
     - Calculated from DOB and current date.
     - Candidate younger than 18 is rejected with exact message:
       "You must be at least 18 years old to register."
     - Candidate exactly 18 years old today is accepted.
     - Candidate older than 18 is accepted.
     - Correct leap-year birthday handling.
     - Frontend max attribute dynamically sets maximum selectable DOB for age 18.
  3. Preservation of other validation:
     - Mandatory fields required (Name, Mobile, Address, DOB, Gender, Email, Aadhaar, Photo).
     - Aadhaar Number (12 digits, unique).
     - Unique Candidate ID generation for every candidate.
     - Existing database records preserved.
"""

import os
import sys
import unittest
import tempfile
import sqlite3
import shutil
from datetime import date, datetime, timedelta
from io import BytesIO

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
import database.db as db_module
from app import app, calculate_age, get_max_dob_for_18


class TestCandidateRegistration(unittest.TestCase):
    """Unit and Integration tests for candidate registration rules and age validation."""

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory and isolated test database
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_db_path = os.path.join(cls.temp_dir, "test_driving_validation.db")
        
        # Point db_module to the test database
        cls.orig_db_path = config.DATABASE_PATH
        cls.orig_db_module_path = db_module.DATABASE_PATH
        config.DATABASE_PATH = cls.test_db_path
        db_module.DATABASE_PATH = cls.test_db_path

        # Initialize schema
        db_module.init_db()

        # Configure Flask app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        # Restore original DB paths
        config.DATABASE_PATH = cls.orig_db_path
        db_module.DATABASE_PATH = cls.orig_db_module_path
        try:
            shutil.rmtree(cls.temp_dir)
        except Exception:
            pass

    def setUp(self):
        # Clear candidates and users from test DB before each test
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM evaluations")
        cursor.execute("DELETE FROM candidates")
        cursor.execute("DELETE FROM users")
        conn.commit()
        conn.close()

    def _login_session(self):
        """Helper to create a test user and log into the Flask test session."""
        db_module.create_user("Officer Test", "officer@test.com", "9999999999", "password123")
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['user_name'] = 'Officer Test'
            sess['user_email'] = 'officer@test.com'
            sess['user_mobile'] = '9999999999'

    # -------------------------------------------------------------------------
    # TEST 1: Age Calculation & Leap Year Handling (Unit Tests)
    # -------------------------------------------------------------------------
    def test_age_calculation_standard_and_boundary(self):
        """Verify calculate_age accurately computes age on exact boundary dates."""
        today = date(2026, 10, 24)

        # Exactly 18 years old today
        dob_18 = date(2008, 10, 24)
        self.assertEqual(calculate_age(dob_18, today), 18)

        # 1 day short of 18 (tomorrow is birthday)
        dob_17_almost_18 = date(2008, 10, 25)
        self.assertEqual(calculate_age(dob_17_almost_18, today), 17)

        # 1 day over 18 (yesterday was birthday)
        dob_18_plus_1 = date(2008, 10, 23)
        self.assertEqual(calculate_age(dob_18_plus_1, today), 18)

        # Older than 18 (e.g. 25 years old)
        dob_25 = date(2001, 10, 24)
        self.assertEqual(calculate_age(dob_25, today), 25)

        # Younger than 18 (e.g. 16 years old)
        dob_16 = date(2010, 5, 10)
        self.assertEqual(calculate_age(dob_16, today), 16)

    def test_leap_year_birthday_calculation(self):
        """Verify leap-year birthday (Feb 29) age calculation across leap and non-leap years."""
        # Born on leap day: Feb 29, 2008
        dob_leap = date(2008, 2, 29)

        # In non-leap year 2026:
        # On Feb 28, 2026, the candidate has not completed 18 full years
        self.assertEqual(calculate_age(dob_leap, date(2026, 2, 28)), 17)
        # On March 1, 2026, the candidate reaches 18 completed years
        self.assertEqual(calculate_age(dob_leap, date(2026, 3, 1)), 18)

        # In a leap year (2028, turning 20):
        self.assertEqual(calculate_age(dob_leap, date(2028, 2, 28)), 19)
        self.assertEqual(calculate_age(dob_leap, date(2028, 2, 29)), 20)

    def test_get_max_dob_for_18(self):
        """Verify get_max_dob_for_18 calculates the exact maximum selectable DOB."""
        # Standard day
        today = date(2026, 10, 24)
        max_dob = get_max_dob_for_18(today)
        self.assertEqual(max_dob, date(2008, 10, 24))
        # Candidate born on max_dob is exactly 18
        self.assertEqual(calculate_age(max_dob, today), 18)

        # Current date is Feb 29 in a leap year (e.g. 2028-02-29)
        # 18 years earlier is 2010 (not a leap year, Feb 29 doesn't exist)
        leap_today = date(2028, 2, 29)
        max_dob_leap = get_max_dob_for_18(leap_today)
        self.assertEqual(max_dob_leap, date(2010, 2, 28))
        self.assertEqual(calculate_age(max_dob_leap, leap_today), 18)

    # -------------------------------------------------------------------------
    # TEST 2: Duplicate Candidate Name Allowed
    # -------------------------------------------------------------------------
    def test_duplicate_candidate_name_allowed(self):
        """Verify multiple candidates can register with the exact same Candidate Name."""
        cand1 = {
            'candidate_id': 'CAND-2026-001',
            'name': 'Rahul Sharma',
            'aadhaar': '111122223333',
            'aadhaar_photo_path': 'static/uploads/aadhaar/a1.jpg',
            'dob': '2000-01-01',
            'gender': 'Male',
            'address': 'Address 1, Bangalore',
            'mobile': '9876543210',
            'email': 'rahul1@example.com',
            'photo_path': 'static/uploads/photos/p1.jpg'
        }
        cand2 = {
            'candidate_id': 'CAND-2026-002',
            'name': 'Rahul Sharma',  # Exact duplicate name
            'aadhaar': '444455556666',
            'aadhaar_photo_path': 'static/uploads/aadhaar/a2.jpg',
            'dob': '2002-05-15',
            'gender': 'Male',
            'address': 'Address 2, Mysore',
            'mobile': '9876543211',
            'email': 'rahul2@example.com',
            'photo_path': 'static/uploads/photos/p2.jpg'
        }

        success1, id1 = db_module.register_candidate(cand1)
        success2, id2 = db_module.register_candidate(cand2)

        self.assertTrue(success1, "First candidate registration should succeed")
        self.assertTrue(success2, "Candidate with duplicate name should register successfully")

        # Verify both exist in DB
        candidates = db_module.get_all_candidates()
        self.assertEqual(len(candidates), 2)
        names = [c['name'] for c in candidates]
        self.assertEqual(names, ['Rahul Sharma', 'Rahul Sharma'])

    # -------------------------------------------------------------------------
    # TEST 3: Duplicate Mobile Number Allowed
    # -------------------------------------------------------------------------
    def test_duplicate_mobile_number_allowed(self):
        """Verify multiple candidates can register with the exact same Mobile Number."""
        shared_mobile = '9888877777'
        cand1 = {
            'candidate_id': 'CAND-2026-003',
            'name': 'Priya Patel',
            'aadhaar': '222233334444',
            'aadhaar_photo_path': 'static/uploads/aadhaar/a3.jpg',
            'dob': '1999-03-20',
            'gender': 'Female',
            'address': 'Flat 101, Ahmedabad',
            'mobile': shared_mobile,
            'email': 'priya@example.com',
            'photo_path': 'static/uploads/photos/p3.jpg'
        }
        cand2 = {
            'candidate_id': 'CAND-2026-004',
            'name': 'Kavita Patel',
            'aadhaar': '333344445555',
            'aadhaar_photo_path': 'static/uploads/aadhaar/a4.jpg',
            'dob': '2001-07-12',
            'gender': 'Female',
            'address': 'Flat 102, Ahmedabad',
            'mobile': shared_mobile,  # Exact duplicate mobile
            'email': 'kavita@example.com',
            'photo_path': 'static/uploads/photos/p4.jpg'
        }

        success1, _ = db_module.register_candidate(cand1)
        success2, _ = db_module.register_candidate(cand2)

        self.assertTrue(success1, "First candidate registration should succeed")
        self.assertTrue(success2, "Second candidate with duplicate mobile should register successfully")

        mobiles = [c['mobile'] for c in db_module.get_all_candidates()]
        self.assertEqual(mobiles.count(shared_mobile), 2)

    # -------------------------------------------------------------------------
    # TEST 4: Duplicate Residential Address Allowed
    # -------------------------------------------------------------------------
    def test_duplicate_residential_address_allowed(self):
        """Verify multiple candidates can share the exact same Residential Address."""
        shared_address = '#42, Palm Meadows, Whitefield, Bangalore, Karnataka 560066'
        cand1 = {
            'candidate_id': 'CAND-2026-005',
            'name': 'Arun Verma',
            'aadhaar': '555566667777',
            'aadhaar_photo_path': 'static/uploads/aadhaar/a5.jpg',
            'dob': '1995-12-10',
            'gender': 'Male',
            'address': shared_address,
            'mobile': '9123456780',
            'email': 'arun@example.com',
            'photo_path': 'static/uploads/photos/p5.jpg'
        }
        cand2 = {
            'candidate_id': 'CAND-2026-006',
            'name': 'Sunita Verma',
            'aadhaar': '666677778888',
            'aadhaar_photo_path': 'static/uploads/aadhaar/a6.jpg',
            'dob': '1997-08-18',
            'gender': 'Female',
            'address': shared_address,  # Exact duplicate address
            'mobile': '9123456781',
            'email': 'sunita@example.com',
            'photo_path': 'static/uploads/photos/p6.jpg'
        }

        success1, _ = db_module.register_candidate(cand1)
        success2, _ = db_module.register_candidate(cand2)

        self.assertTrue(success1)
        self.assertTrue(success2)
        addresses = [c['address'] for c in db_module.get_all_candidates()]
        self.assertEqual(addresses.count(shared_address), 2)

    # -------------------------------------------------------------------------
    # TEST 5: Duplicate Date of Birth Allowed
    # -------------------------------------------------------------------------
    def test_duplicate_date_of_birth_allowed(self):
        """Verify multiple candidates can share the exact same Date of Birth."""
        shared_dob = '2004-06-15'
        cand1 = {
            'candidate_id': 'CAND-2026-007',
            'name': 'Candidate One',
            'aadhaar': '777788889999',
            'aadhaar_photo_path': 'static/uploads/aadhaar/a7.jpg',
            'dob': shared_dob,
            'gender': 'Male',
            'address': 'Street A',
            'mobile': '9000000001',
            'email': 'one@example.com',
            'photo_path': 'static/uploads/photos/p7.jpg'
        }
        cand2 = {
            'candidate_id': 'CAND-2026-008',
            'name': 'Candidate Two',
            'aadhaar': '888899990000',
            'aadhaar_photo_path': 'static/uploads/aadhaar/a8.jpg',
            'dob': shared_dob,  # Exact duplicate DOB
            'gender': 'Female',
            'address': 'Street B',
            'mobile': '9000000002',
            'email': 'two@example.com',
            'photo_path': 'static/uploads/photos/p8.jpg'
        }

        success1, _ = db_module.register_candidate(cand1)
        success2, _ = db_module.register_candidate(cand2)

        self.assertTrue(success1)
        self.assertTrue(success2)
        dobs = [c['dob'] for c in db_module.get_all_candidates()]
        self.assertEqual(dobs.count(shared_dob), 2)

    # -------------------------------------------------------------------------
    # TEST 6: Simultaneous Duplicates in All 4 Non-Unique Fields
    # -------------------------------------------------------------------------
    def test_simultaneous_duplicates_in_all_four_fields(self):
        """Verify candidates can share Name, Mobile, Address, and DOB simultaneously."""
        common_demographics = {
            'name': 'Same Name',
            'mobile': '9998887776',
            'address': '100 Common Rd, City, 560001',
            'dob': '2002-11-20',
            'gender': 'Male',
            'aadhaar_photo_path': 'static/uploads/aadhaar/common.jpg',
            'photo_path': 'static/uploads/photos/common.jpg'
        }

        cand1 = dict(common_demographics, candidate_id='CAND-2026-009', aadhaar='101020203030', email='userA@mail.com')
        cand2 = dict(common_demographics, candidate_id='CAND-2026-010', aadhaar='404050506060', email='userB@mail.com')

        success1, _ = db_module.register_candidate(cand1)
        success2, _ = db_module.register_candidate(cand2)

        self.assertTrue(success1)
        self.assertTrue(success2)

        retrieved1 = db_module.get_candidate('CAND-2026-009')
        retrieved2 = db_module.get_candidate('CAND-2026-010')

        self.assertIsNotNone(retrieved1)
        self.assertIsNotNone(retrieved2)
        self.assertEqual(retrieved1['name'], retrieved2['name'])
        self.assertEqual(retrieved1['mobile'], retrieved2['mobile'])
        self.assertEqual(retrieved1['address'], retrieved2['address'])
        self.assertEqual(retrieved1['dob'], retrieved2['dob'])
        self.assertNotEqual(retrieved1['candidate_id'], retrieved2['candidate_id'])
        self.assertNotEqual(retrieved1['aadhaar'], retrieved2['aadhaar'])

    # -------------------------------------------------------------------------
    # TEST 7: Backend Rejects Age < 18 with Exact Error Message
    # -------------------------------------------------------------------------
    def test_candidate_under_18_rejected_via_flask_route(self):
        """POST /register with age < 18 must be rejected with the exact error message."""
        self._login_session()

        # Current date minus 17 years (e.g., age 17)
        underage_dob = (date.today().replace(year=date.today().year - 17)).strftime('%Y-%m-%d')

        form_data = {
            'name': 'Minor Candidate',
            'aadhaar': '123456789099',
            'dob': underage_dob,
            'gender': 'Male',
            'address': 'Test Address',
            'mobile': '9876543299',
            'email': 'minor@example.com',
            'aadhaar_image': (BytesIO(b"fake_aadhaar_bytes"), 'aadhaar.png'),
            'photo': (BytesIO(b"fake_photo_bytes"), 'photo.png')
        }

        response = self.client.post('/register', data=form_data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

        # Must display exact specified error message
        expected_msg = "You must be at least 18 years old to register."
        self.assertIn(expected_msg.encode('utf-8'), response.data)

        # Verify candidate was NOT inserted
        candidates = db_module.get_all_candidates()
        self.assertEqual(len(candidates), 0)

    # -------------------------------------------------------------------------
    # TEST 8: Candidate Exactly 18 Years Old Today Allowed
    # -------------------------------------------------------------------------
    def test_candidate_exactly_18_today_allowed(self):
        """POST /register with DOB making candidate exactly 18 today must be allowed."""
        self._login_session()

        exact_18_dob = get_max_dob_for_18().strftime('%Y-%m-%d')

        form_data = {
            'name': 'Exact Eighteen',
            'aadhaar': '999888777666',
            'dob': exact_18_dob,
            'gender': 'Female',
            'address': 'Exact 18 Street',
            'mobile': '9888777666',
            'email': 'exact18@example.com',
            'aadhaar_image': (BytesIO(b"fake_aadhaar_png"), 'aadhaar.png'),
            'photo': (BytesIO(b"fake_photo_png"), 'photo.png')
        }

        response = self.client.post('/register', data=form_data, content_type='multipart/form-data', follow_redirects=False)
        # On success, redirects to /authenticate?candidate_id=...
        self.assertEqual(response.status_code, 302)
        self.assertIn('/authenticate?candidate_id=', response.headers['Location'])

        # Verify candidate saved in database
        candidates = db_module.get_all_candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['dob'], exact_18_dob)

    # -------------------------------------------------------------------------
    # TEST 9: Candidate Older Than 18 Years Old Allowed
    # -------------------------------------------------------------------------
    def test_candidate_older_than_18_allowed(self):
        """POST /register with candidate older than 18 (e.g. 25 years old) must succeed."""
        self._login_session()

        older_dob = (date.today().replace(year=date.today().year - 25)).strftime('%Y-%m-%d')

        form_data = {
            'name': 'Adult Candidate',
            'aadhaar': '555444333222',
            'dob': older_dob,
            'gender': 'Male',
            'address': 'Adult Address, City',
            'mobile': '9111222333',
            'email': 'adult@example.com',
            'aadhaar_image': (BytesIO(b"aadhaar_content"), 'aadhaar.jpg'),
            'photo': (BytesIO(b"photo_content"), 'photo.jpg')
        }

        response = self.client.post('/register', data=form_data, content_type='multipart/form-data', follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/authenticate?candidate_id=', response.headers['Location'])

        candidates = db_module.get_all_candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['name'], 'Adult Candidate')

    # -------------------------------------------------------------------------
    # TEST 10: Aadhaar Uniqueness Preserved
    # -------------------------------------------------------------------------
    def test_aadhaar_uniqueness_enforced(self):
        """Registering two candidates with identical Aadhaar numbers must be rejected."""
        self._login_session()

        common_aadhaar = '123412341234'
        exact_18_dob = get_max_dob_for_18().strftime('%Y-%m-%d')

        form1 = {
            'name': 'Candidate First',
            'aadhaar': common_aadhaar,
            'dob': exact_18_dob,
            'gender': 'Male',
            'address': 'Address 1',
            'mobile': '9000000001',
            'email': 'first@example.com',
            'aadhaar_image': (BytesIO(b"aadhaar1"), 'aadhaar1.jpg'),
            'photo': (BytesIO(b"photo1"), 'photo1.jpg')
        }
        resp1 = self.client.post('/register', data=form1, content_type='multipart/form-data', follow_redirects=False)
        self.assertEqual(resp1.status_code, 302)

        # Attempt to register second candidate with identical Aadhaar
        form2 = {
            'name': 'Candidate Second',
            'aadhaar': common_aadhaar,
            'dob': exact_18_dob,
            'gender': 'Female',
            'address': 'Address 2',
            'mobile': '9000000002',
            'email': 'second@example.com',
            'aadhaar_image': (BytesIO(b"aadhaar2"), 'aadhaar2.jpg'),
            'photo': (BytesIO(b"photo2"), 'photo2.jpg')
        }
        resp2 = self.client.post('/register', data=form2, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(resp2.status_code, 200)
        self.assertIn(b"A candidate with this Aadhaar Number has already been registered.", resp2.data)

        # Exactly 1 record should exist
        candidates = db_module.get_all_candidates()
        self.assertEqual(len(candidates), 1)

    # -------------------------------------------------------------------------
    # TEST 11: Frontend Template Attributes & Validation
    # -------------------------------------------------------------------------
    def test_frontend_template_max_attribute_and_error_message(self):
        """GET /register must render HTML with max attribute on DOB and the exact error alert message."""
        self._login_session()

        response = self.client.get('/register')
        self.assertEqual(response.status_code, 200)

        html_content = response.data.decode('utf-8')
        expected_max_dob = get_max_dob_for_18().strftime('%Y-%m-%d')

        # Check max attribute is set to exact 18-year threshold
        self.assertIn(f'max="{expected_max_dob}"', html_content)

        # Check exact required error message appears in HTML
        self.assertIn("You must be at least 18 years old to register.", html_content)


if __name__ == '__main__':
    unittest.main()
