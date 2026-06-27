import json, glob, requests

folder = r"D:\youssef hefzy\College\Graduation project\current work\EmailHeaderAnomalyDetection_Extended\Phishing Pot\Test_24"
files = sorted(glob.glob(folder + "/*.json"))
print(f"Testing {len(files)} emails...")
print()

legit_count = 0
mal_count = 0

for f in files:
    with open(f, "r") as fp:
        data = json.load(fp)
    
    payload = {
        "email_id": data["email_id"],
        "headers": data.get("headers", {}),
        "header_features": data["header_features"],
        "body_text": "",
        "attachments": []
    }
    
    try:
        resp = requests.post("http://127.0.0.1:8001/predict-header", json=payload, timeout=10)
        result = resp.json()
        
        pred = result["prediction"]
        conf = result["confidence"]
        
        if pred == "legitimate":
            legit_count += 1
            symbol = "O"
        else:
            mal_count += 1
            symbol = "X"
        
        subject = data["header_features"].get("subject_len", "?")
        base64 = data["header_features"].get("is_base64_encoded", "?")
        spf = data["header_features"].get("spf_result", "?")
        
        print(symbol + " " + data["email_id"] + ": " + pred + " (conf: " + str(round(conf, 4)) + ") | subject=" + str(subject) + ", base64=" + str(base64) + ", spf=" + str(spf))
    except Exception as e:
        print("ERROR " + data["email_id"] + ": " + str(e))

print()
print("Results: " + str(mal_count) + " malicious, " + str(legit_count) + " legitimate out of " + str(len(files)))
detection_rate = 100 * mal_count / len(files) if len(files) > 0 else 0
print("Detection rate: " + str(mal_count) + "/" + str(len(files)) + " (" + str(round(detection_rate, 1)) + "%)")
