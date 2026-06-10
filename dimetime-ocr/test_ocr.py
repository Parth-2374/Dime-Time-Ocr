import urllib.request
import urllib.parse
import json
import mimetypes

def send_multipart_request(url, fields, files, headers=None):
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    data = []
    
    for name, value in fields.items():
        data.append(f'--{boundary}')
        data.append(f'Content-Disposition: form-data; name="{name}"')
        data.append('')
        data.append(str(value))
        
    for name, filename in files.items():
        data.append(f'--{boundary}')
        mimetype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        data.append(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"')
        data.append(f'Content-Type: {mimetype}')
        data.append('')
        with open(filename, 'rb') as f:
            data.append(f.read())
            
    data.append(f'--{boundary}--')
    data.append('')
    
    body = b''
    for item in data:
        if isinstance(item, bytes):
            body += item + b'\r\n'
        else:
            body += item.encode('utf-8') + b'\r\n'
            
    req = urllib.request.Request(url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
            
    try:
        with urllib.request.urlopen(req) as res:
            return res.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        return e.read().decode('utf-8')

# 1. Query Python OCR
print("--- Querying Python OCR Service ---")
try:
    ocr_res = send_multipart_request("http://localhost:8000/ocr/mtc-extract", {}, {"file": "steel_plate_tag.png"})
    print(ocr_res)
except Exception as e:
    print(f"Error: {e}")

# 2. Query Java backend (login first)
print("\n--- Querying Java Backend ---")
try:
    login_data = json.dumps({"username": "admin", "password": "admin123"}).encode('utf-8')
    req = urllib.request.Request("http://localhost:8080/api/auth/login", data=login_data)
    req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req) as res:
        token = json.loads(res.read().decode('utf-8')).get("token")
        
    headers = {"Authorization": f"Bearer {token}"}
    upload_res = send_multipart_request(
        "http://localhost:8080/api/mtc-documents/ocr-upload",
        {"uploadedBy": "admin", "poNumber": "PO-2026-013"},
        {"file": "steel_plate_tag.png"},
        headers
    )
    print(upload_res)
except Exception as e:
    print(f"Error: {e}")
