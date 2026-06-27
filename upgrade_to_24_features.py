#!/usr/bin/env python3
"""upgrade_to_24_features.py - Add G5 authentication features to 16-feature JSONs."""

import json, sys, os
from pathlib import Path

def upgrade_json(input_path, output_path=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data.get('header_features', {})
    
    # Check if already has 24 features
    if 'spf_result' in features:
        print(f"  SKIP {input_path.name} - already has 24 features")
        return data
    
    # Get date to determine era
    date_str = data.get('headers', {}).get('date', '')
    is_modern = False
    if date_str:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            is_modern = dt.year >= 2010
        except Exception:
            pass
    
    if is_modern:
        # Modern email - default to no auth (might be missing)
        features['spf_result'] = 0
        features['dkim_result'] = 0
        features['dmarc_result'] = 0
        features['dkim_signature_present'] = 0
        features['dkim_signature_valid_format'] = 0
        features['auth_results_present'] = 0
        features['all_auth_pass'] = 0
        features['any_auth_fail'] = 0
    else:
        # Old email - no auth available
        features['spf_result'] = 0
        features['dkim_result'] = 0
        features['dmarc_result'] = 0
        features['dkim_signature_present'] = 0
        features['dkim_signature_valid_format'] = 0
        features['auth_results_present'] = 0
        features['all_auth_pass'] = 0
        features['any_auth_fail'] = 0
    
    data['header_features'] = features
    
    if output_path is None:
        output_path = input_path
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return data

def upgrade_folder(input_folder, output_folder=None):
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Folder not found: {input_path}")
        return
    
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    json_files = sorted(input_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    upgraded = 0
    skipped = 0
    
    for jf in json_files:
        out = Path(output_folder) / jf.name if output_folder else None
        result = upgrade_json(jf, out)
        if 'spf_result' in result.get('header_features', {}):
            upgraded += 1
        else:
            skipped += 1
    
    print(f"Done. Upgraded: {upgraded}, Skipped: {skipped}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python upgrade_to_24_features.py input.json [output.json]")
        print("  Folder:      python upgrade_to_24_features.py input_folder/ -o output_folder/")
        sys.exit(1)
    
    input_arg = sys.argv[1]
    
    # Parse -o flag
    output_arg = None
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            output_arg = sys.argv[idx + 1]
    
    input_path = Path(input_arg)
    
    if input_path.is_dir():
        upgrade_folder(input_path, output_arg)
    else:
        result = upgrade_json(input_path, output_arg)
        print(json.dumps(result, indent=2))
