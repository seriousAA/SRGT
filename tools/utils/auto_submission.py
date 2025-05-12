import requests
from bs4 import BeautifulSoup
import time
import json
from pathlib import Path
from datetime import datetime

success = 0

# Configuration
login_url = 'http://bed4rs.net:8005/login/'
login_url_with_query = login_url + '?next=/evaluation1/'
submission_url = 'http://bed4rs.net:8005/evaluation1/'
delay_between_submissions = 30  # Delay in seconds between each submission

def login(session, username, password):
    # Step 1: Get the CSRF token from the login page
    response = session.get(login_url_with_query)
    response = session.get(login_url_with_query)
    soup = BeautifulSoup(response.text, 'html.parser')
    csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']
    
    # Step 2: Prepare the form data
    login_data = {
        'csrfmiddlewaretoken': csrf_token,
        'next': '/evaluation1/',  # 'next' in both query and form data
        'username': username,
        'password': password
    }
    
    # Step 3: Prepare the headers
    headers = {
        'Referer': login_url_with_query,
        'Origin': 'http://bed4rs.net:8005',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'keep-alive'
    }
    
    # Step 4: Perform the login request
    response = session.post(login_url, data=login_data, headers=headers)
    
    # Check if login was successful by verifying the response and cookies
    if 'sessionid' in session.cookies:
        print(f"Login successful for user {username}!")
        soup = BeautifulSoup(response.text, 'html.parser')
        csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']
        return csrf_token
    else:
        print(f"Login failed for user {username}!")
        return None

def logout(session):
    session.cookies.clear()
    print("Logged out and cleared session.")

def load_users_and_submit(json_file, submission_files):
    print(f"[{datetime.now().strftime('%Y.%m.%d %H:%M:%S')}] Starting submission process...")
    # Load user credentials from JSON file
    with open(json_file, 'r') as f:
        users = json.load(f)['users']
    
    assert len(users) == len(submission_files)

    for i, user in enumerate(users):
        with requests.Session() as session:
            session.proxies = {
                'http': None,
                'https': None
            }
            csrf_token = login(session, user['username'], user['password'])
            if csrf_token:
                submit_file(session, submission_files[i], csrf_token)
                logout(session)
                if i < len(users) - 1:
                    print("-----------Loading------------")
                    time.sleep(delay_between_submissions)
            else:
                print(f"Skipping submission for {user['username']} due to login failure.")
    
    print(f"All files have been submitted, success: {success}")

def submit_file(session, file_path, csrf_token):
    global success
    file_name = Path(file_path).name
    description = Path(file_path).stem
    print(f"Submitting file: {file_name}")
    
    # Get CSRF token from the submission page
    response = session.get(submission_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']

    with open(file_path, 'rb') as f:
        files = {
            'docfile': (file_name, f)
        }
        data = {
            'csrfmiddlewaretoken': csrf_token,
            'description': description  # Set the description to the file basename without extension
        }
        
        headers = {
            'Referer': submission_url,
            'Origin': 'http://bed4rs.net:8005',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive'
        }
        
        response = session.post(submission_url, files=files, data=data, headers=headers, timeout=1200)

        request_duration = response.elapsed.total_seconds() / 60.0  # Convert to minutes
        
        # Extract the content length from the response headers
        content_length = int(response.headers.get('Content-Length', 'Unknown'))

        if response.status_code == 200:
            success += 1
            if content_length >= 8000:
                print(f"Submission successful for {file_name}")
            else:
                print("Something went wrong, please check your email for detail.")
        else:
            print(f"Submission failed for {file_name}. Status code: {response.status_code}")
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the textarea containing the traceback
            traceback_area = soup.find('textarea', {'id': 'traceback_area'})
            
            if traceback_area:
                # Extract and print the core traceback information
                traceback_text = traceback_area.get_text()
                print(f"Error traceback details: \n{traceback_text.strip()}")
            else:
                print(f"Error message: {response.text}")

        # Print the required information
        print(f"File: {file_name}")
        print(f"Time duration: {request_duration:.2f} minute(s)")
        print(f"Response Content-Length: {content_length}")

            
if __name__ == "__main__":
    json_file = 'tools/utils/submission_users.json'  # Replace with the path to your JSON file
    folder_name = "results"
    basenames = [
        "partial_semi_detr/2_1_semi_detr_1215/2_1_semi_detr_1215.zip",
        "partial_semi_detr/3_1_semi_detr_1219/3_1_semi_detr_1219.zip",
        "partial_semi_detr/4_1_semi_detr_1222/4_1_semi_detr_1222.zip",
        "partial_semi_detr/5_1_semi_detr_1224/5_1_semi_detr_1224.zip",
        "partial_semi_detr/6_1_semi_detr_1226/6_1_semi_detr_1226.zip",
        "partial_semi_detr/7_1_semi_detr_0104/7_1_semi_detr_0104.zip",
        "partial_semi_detr/8_1_semi_detr_0105/8_1_semi_detr_0105.zip",
        "partial_semi_detr/9_1_semi_detr_0107/9_1_semi_detr_0107.zip",
        "partial_semi_detr/10_1_semi_detr_0110/10_1_semi_detr_0110.zip"
    ]

    # Generate the list of file paths
    submission_files = [f"{folder_name}/{basename}" for basename in basenames]
    
    load_users_and_submit(json_file, submission_files)