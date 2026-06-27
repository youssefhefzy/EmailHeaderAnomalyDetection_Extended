import json, glob, re, os
from email.parser import BytesParser
from email import policy
from pathlib import Path

def parse_auth_result(value):
    if not value:
        return 0
    v = value.lower().strip()
    if "pass" in v: return 1
    if any(w in v for w in ["fail","softfail","hardfail","none"]): return -1
    if any(w in v for w in ["neutral","temperror","permerror"]): return 0
    return 0

def extract_auth_from_eml(eml_path):
    with open(eml_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    
    auth_header = msg.get("Authentication-Results", "")
    if not auth_header:
        auth_header = str(msg.get("Authentication-Results", ""))
    
    spf_match = re.search(r'spf\s*=\s*(\S+)', str(auth_header), re.IGNORECASE)
    dkim_match = re.search(r'dkim\s*=\s*(\S+)', str(auth_header), re.IGNORECASE)
    dmarc_match = re.search(r'dmarc\s*=\s*(\S+)', str(auth_header), re.IGNORECASE)
    
    return {
        "spf_result": parse_auth_result(spf_match.group(1)) if spf_match else 0,
        "dkim_result": parse_auth_result(dkim_match.group(1)) if dkim_match else 0,
        "dmarc_result": parse_auth_result(dmarc_match.group(1)) if dmarc_match else 0,
        "auth_results_present": int(bool(auth_header)),
        "any_auth_fail": 1 if any(
            parse_auth_result(m.group(1)) == -1 
            for m in [spf_match, dkim_match, dmarc_match] if m
        ) else 0,
        "all_auth_pass": 1 if all(
            parse_auth_result(m.group(1)) == 1 
            for m in [spf_match, dkim_match, dmarc_match] if m
        ) else 0,
    }

def fix_json_folder(json_folder, eml_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    json_files = sorted(glob.glob(json_folder + "/*.json"))
    eml_files = {f.stem: f for f in Path(eml_folder).glob("*.eml")}
    
    fixed = 0
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        
        email_id = data.get("email_id", "")
        msg_id = data.get("headers", {}).get("message_id", "")
        
        # Try to find matching .eml
        eml_path = None
        for stem, path in eml_files.items():
            with open(path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
            if msg_id and msg_id in content:
                eml_path = path
                break
        
        if eml_path:
            auth = extract_auth_from_eml(eml_path)
            data["header_features"].update(auth)
            fixed += 1
        
        out_path = os.path.join(output_folder, os.path.basename(jf))
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"Fixed {fixed}/{len(json_files)} files with auth data")

if __name__ == "__main__":
    import sys
    fix_json_folder(sys.argv[1], sys.argv[2], sys.argv[3])
