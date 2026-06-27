#!/usr/bin/env python3
"""extract_eml_folder_header_features.py - Extract 24 header features from email files."""

from __future__ import annotations
import argparse, json, re
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Optional

def decode_mime_value(value: Optional[str]) -> str:
    if not value: return ""
    try: return str(make_header(decode_header(value))).strip()
    except: return str(value).strip()

def parse_header_date(date_value: str) -> bool:
    if not date_value: return False
    try:
        parsedate_to_datetime(date_value)
        return True
    except: return False

def is_message_id_malformed(message_id: str) -> int:
    if not message_id: return 0
    return int(not re.match(r"^<[^<>@\s]+@[^<>@\s]+>$", message_id.strip()))

def parse_auth_result(result_value: Optional[str]) -> int:
    if not result_value: return 0
    v = result_value.lower().strip()
    if "pass" in v: return 1
    if any(w in v for w in ["fail","softfail","hardfail","none","temperror","permerror"]): return -1
    if "neutral" in v: return 0
    return 0

def extract_authentication_results(message: Message) -> Dict[str, Any]:
    auth = decode_mime_value(message.get("Authentication-Results"))
    auth_present = int(bool(auth))
    spf, dkim, dmarc = 0, 0, 0
    if auth:
        spf_m = re.search(r'spf\s*=\s*(\S+)', auth, re.IGNORECASE)
        dkim_m = re.search(r'dkim\s*=\s*(\S+)', auth, re.IGNORECASE)
        dmarc_m = re.search(r'dmarc\s*=\s*(\S+)', auth, re.IGNORECASE)
        spf = parse_auth_result(spf_m.group(1)) if spf_m else 0
        dkim = parse_auth_result(dkim_m.group(1)) if dkim_m else 0
        dmarc = parse_auth_result(dmarc_m.group(1)) if dmarc_m else 0
    dkim_sig = message.get("DKIM-Signature")
    dkim_present = int(dkim_sig is not None)
    dkim_valid = 0
    if dkim_sig:
        dkim_str = decode_mime_value(dkim_sig)
        if all(t in dkim_str for t in ["v=","a=","b=","bh=","d=","h=","s="]):
            dkim_valid = 1
    all_pass = 1 if (auth_present and spf==1 and dkim==1 and dmarc==1) else 0
    any_fail = 1 if (spf==-1 or dkim==-1 or dmarc==-1) else 0
    return {
        "spf_result": spf, "dkim_result": dkim, "dmarc_result": dmarc,
        "dkim_signature_present": dkim_present,
        "dkim_signature_valid_format": dkim_valid,
        "auth_results_present": auth_present,
        "all_auth_pass": all_pass, "any_auth_fail": any_fail,
    }

def is_email_file(file_path: Path) -> bool:
    try:
        with file_path.open("rb") as f:
            first = f.readline().decode("utf-8",errors="ignore").strip().lower()
            return any(first.startswith(p) for p in [
                "from ","received:","return-path:","date:","message-id:",
                "delivered-to:","x-","subject:","to:","mime-version:",
                "content-type:","content-transfer-encoding:"
            ])
    except: return False

def load_eml(file_path: Path) -> Message:
    with file_path.open("rb") as f:
        return BytesParser(policy=policy.default).parse(f)

def extract_headers(message: Message) -> Dict[str, str]:
    return {
        "date": decode_mime_value(message.get("Date")),
        "message_id": decode_mime_value(message.get("Message-ID")),
    }

def extract_header_features(message: Message) -> Dict[str, Any]:
    subject = decode_mime_value(message.get("Subject"))
    date_value = decode_mime_value(message.get("Date"))
    message_id = decode_mime_value(message.get("Message-ID"))
    cte = decode_mime_value(message.get("Content-Transfer-Encoding")).lower()
    received = message.get_all("Received") or []
    features = {
        "has_message_id": int(message.get("Message-ID") is not None),
        "has_date": int(message.get("Date") is not None),
        "has_from": int(message.get("From") is not None),
        "has_to": int(message.get("To") is not None),
        "has_subject": int(message.get("Subject") is not None),
        "has_content_type": int(message.get("Content-Type") is not None),
        "has_mime_version": int(message.get("MIME-Version") is not None),
        "has_reply_to": int(message.get("Reply-To") is not None),
        "has_x_mailer": int(message.get("X-Mailer") is not None),
        "has_x_originating_ip": int(message.get("X-Originating-IP") is not None),
        "subject_len": len(subject),
        "n_received_hops": len(received),
        "is_base64_encoded": int("base64" in cte),
        "malformed_date": int(bool(date_value) and not parse_header_date(date_value)),
        "mid_malformed": is_message_id_malformed(message_id),
        "has_x_priority": int(message.get("X-Priority") is not None),
    }
    auth = extract_authentication_results(message)
    features.update(auth)
    return features

def build_output(file_path: Path, email_id: str) -> Dict[str, Any]:
    msg = load_eml(file_path)
    return {
        "email_id": email_id,
        "headers": extract_headers(msg),
        "header_features": extract_header_features(msg),
        "body_text": "",
        "attachments": [],
    }

def extract_folder(input_folder, output_folder, recursive=False, start_index=1, max_files=0):
    inp = Path(input_folder)
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in (inp.rglob("*") if recursive else inp.glob("*")) if f.is_file()])
    eml = [f for f in files if is_email_file(f)]
    if not eml:
        print(f"No email files found in {inp}")
        return
    if max_files > 0: eml = eml[:max_files]
    print(f"Found {len(eml)} email files out of {len(files)} total")
    ok, fail = 0, 0
    for i, fp in enumerate(eml, start=start_index):
        eid = f"email_{i:04d}"
        try:
            res = build_output(fp, eid)
            with (out / f"{eid}.json").open("w",encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            ok += 1
            if ok % 100 == 0: print(f"[OK] {ok} files...")
        except Exception as e:
            fail += 1
            if fail <= 5: print(f"[FAIL] {fp.name}: {e}")
    print(f"Done. OK: {ok}, Failed: {fail}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_folder")
    p.add_argument("--output","-o",default="header_json_output")
    p.add_argument("--recursive",action="store_true")
    p.add_argument("--start-index",type=int,default=1)
    p.add_argument("--max-files",type=int,default=0)
    args = p.parse_args()
    extract_folder(args.input_folder, args.output, args.recursive, args.start_index, args.max_files)

if __name__ == "__main__":
    main()
